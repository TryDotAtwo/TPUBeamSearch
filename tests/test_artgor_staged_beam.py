import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from tpu_beam_search.artgor_exact_inference import (
    ArtgorExactConfig,
    ArtgorExactInference,
)

from tpu_beam_search.artgor_staged_beam import (
    ArtgorExactBeamRuntime,
    StagedDepthConfig,
    StagedDepthExecutables,
    beam_solve_v_only_spmd_packed_exact,
    build_staged_depth_executables,
    concatenate_q_chunks,
    inference_chunk_starts,
    parent_window_starts,
    run_staged_depth,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "third_party" / "artgor_cube555_v344319112"


def _load_artgor_beam_module():
    sys.path.insert(0, str(SNAPSHOT))
    try:
        spec = importlib.util.spec_from_file_location(
            "_artgor_beam_depth_test",
            SNAPSHOT / "jax_beam_spmd_v_only.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SNAPSHOT))


def _oracle_q(_params, states, dtype=jnp.bfloat16):
    states = states.astype(jnp.float32)
    coeff = jnp.arange(1, states.shape[1] + 1, dtype=jnp.float32)
    first = jnp.sum(states * coeff, axis=1) * jnp.float32(0.125)
    second = jnp.sum(states * coeff[::-1], axis=1) * jnp.float32(0.0625)
    return jnp.stack((first, second), axis=1).astype(dtype)


def _named_outputs(outputs, *, n_gen):
    names = (
        "states",
        "packed_backpointer",
        "min_v_log",
        "found_step",
        "found_pos_local",
        "found_pos_rank",
        "verify_state",
        "history",
        "in_move",
        "trace_hit",
        "trace_pos",
        "trace_v",
        "trace_cutoff",
    )
    named = {name: np.asarray(value) for name, value in zip(names, outputs)}
    packed = named["packed_backpointer"].astype(np.uint32)
    named["owners"] = ((packed >> np.uint32(24)) & np.uint32(0x7)).astype(
        np.int32
    )
    parent_local = (packed & np.uint32((1 << 24) - 1)).astype(np.int64)
    move = ((packed >> np.uint32(27)) & np.uint32(0x1F)).astype(np.int64)
    named["selected_ids"] = parent_local * n_gen + move
    named["scores"] = named["min_v_log"]
    return named


def test_default_depth_geometry_has_64_inference_chunks_and_16_search_windows():
    config = StagedDepthConfig(
        world_size=8,
        b_local=2_097_152,
        inference_chunk=32_768,
        parent_chunk=131_072,
        n_gen=30,
        state_size=150,
    )
    config.validate()

    assert len(inference_chunk_starts(config)) == 64
    assert inference_chunk_starts(config)[:4] == (
        0,
        32_768,
        65_536,
        98_304,
    )
    assert inference_chunk_starts(config)[-1] == 2_064_384
    assert len(parent_window_starts(config)) == 16
    assert parent_window_starts(config)[-1] == 1_966_080


def test_q_chunks_preserve_parent_then_move_order():
    parts = [
        np.full((1, 2, 3), value, np.float32) for value in range(4)
    ]
    actual = np.asarray(concatenate_q_chunks(parts))

    assert actual.shape == (1, 8, 3)
    np.testing.assert_array_equal(
        actual[0, :, 0], [0, 0, 1, 1, 2, 2, 3, 3]
    )


def test_small_geometry_is_available_only_for_diagnostics():
    config = StagedDepthConfig(
        world_size=1,
        b_local=8,
        inference_chunk=2,
        parent_chunk=4,
        n_gen=2,
        state_size=4,
    )
    config.validate(require_published_geometry=False)
    with np.testing.assert_raises(ValueError):
        config.validate()


def test_run_staged_depth_assembles_every_q_chunk_in_parent_order():
    config = StagedDepthConfig(
        world_size=1,
        b_local=8,
        inference_chunk=2,
        parent_chunk=4,
        n_gen=2,
        state_size=4,
    )
    states = jnp.arange(32, dtype=jnp.uint8).reshape(1, 8, 4)

    def prefix_from_beam(full_states, _weights, start):
        return jax.lax.dynamic_slice(
            full_states,
            (jnp.int32(0), start, jnp.int32(0)),
            (1, 2, 4),
        )

    def head(hidden, _weights):
        row_id = hidden[:, :, 0].astype(jnp.bfloat16)
        return jnp.stack((row_id, row_id + 1), axis=2)

    def search_depth(full_states, q_values, *carry):
        return full_states, q_values, carry

    executables = StagedDepthExecutables(
        config=config,
        prefix_from_beam=prefix_from_beam,
        head=head,
        assemble_q=concatenate_q_chunks,
        search_depth=search_depth,
    )
    actual_states, actual_q, actual_carry = run_staged_depth(
        executables, states, object(), "carry", jnp.int32(1)
    )

    np.testing.assert_array_equal(actual_states, states)
    np.testing.assert_array_equal(
        actual_q[0, :, 0], np.arange(0, 32, 4, dtype=np.float32)
    )
    assert actual_carry == ("carry", jnp.int32(1))


def test_precomputed_q_depth_matches_original_depth_tensor_for_tensor():
    artgor = _load_artgor_beam_module()
    artgor.model_apply = _oracle_q

    world_size = 1
    b_local = 8
    parent_chunk = 4
    n_gen = 2
    state_size = 4
    k_per_peer = 8
    num_steps = 4
    all_moves = jnp.asarray(
        [[1, 0, 2, 3], [0, 2, 1, 3]], dtype=jnp.int32
    )
    states = jnp.asarray(
        [[
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [0, 2, 1, 3],
            [2, 0, 1, 3],
            [1, 2, 0, 3],
            [2, 1, 0, 3],
            [0, 1, 3, 2],
            [1, 0, 3, 2],
        ]],
        dtype=jnp.uint8,
    )
    v0 = jnp.asarray([3, 2, 1, 0], dtype=jnp.uint8)
    hash_vec = jnp.asarray([17, 31, 47, 73], dtype=jnp.int64)
    v0_hash = jnp.sum(v0.astype(jnp.int64) * hash_vec)
    min_v_log = jnp.full((1, num_steps), 1e6, dtype=jnp.float32)
    found_step = jnp.full((1,), -1, dtype=jnp.int32)
    found_pos_local = jnp.full((1,), -1, dtype=jnp.int32)
    found_pos_rank = jnp.full((1,), -1, dtype=jnp.int32)
    verify_state = jnp.zeros((1, state_size), dtype=jnp.uint8)
    history = jnp.full((1, 1, b_local), -1, dtype=jnp.int64)
    in_move = jnp.full((1, b_local), -1, dtype=jnp.int8)

    original_body = artgor._build_step_body_v_only_packed_streaming(
        None,
        all_moves,
        v0,
        hash_vec,
        v0_hash,
        b_local,
        world_size,
        k_per_peer,
        n_gen,
        state_size,
        jnp.bfloat16,
        2,
        parent_chunk=parent_chunk,
        pack_v_score=True,
        history_depth=1,
        history_exact=True,
        q_mode=True,
    )
    original_mesh = Mesh(np.asarray(jax.devices()[:1]), ("cores",))
    staged_mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))

    @jax.jit
    def run_original(*args):
        return jax.shard_map(
            original_body,
            mesh=original_mesh,
            in_specs=(P("cores"),) * 8 + (P(),),
            out_specs=(P("cores"),) * 13,
            check_vma=False,
        )(*args)

    exact_inference = SimpleNamespace(
        prefix=lambda chunk, _weights: chunk,
        head=lambda hidden, _weights: _oracle_q(
            None, hidden[0], dtype=jnp.bfloat16
        )[None],
    )
    executables = build_staged_depth_executables(
        exact_inference,
        None,
        mesh=staged_mesh,
        config=StagedDepthConfig(
            world_size=world_size,
            b_local=b_local,
            inference_chunk=2,
            parent_chunk=parent_chunk,
            n_gen=n_gen,
            state_size=state_size,
        ),
        all_moves=all_moves,
        V0=v0,
        hash_vec=hash_vec,
        V0_hash=v0_hash,
        K_per_peer=k_per_peer,
        history_depth=1,
        history_exact=True,
        donate_search_carry=False,
        require_published_geometry=False,
    )

    original = _named_outputs(
        run_original(
            states,
            min_v_log,
            found_step,
            found_pos_local,
            found_pos_rank,
            verify_state,
            history,
            in_move,
            jnp.int32(1),
        ),
        n_gen=n_gen,
    )
    staged = _named_outputs(
        run_staged_depth(
            executables,
            states,
            None,
            min_v_log,
            found_step,
            found_pos_local,
            found_pos_rank,
            verify_state,
            history,
            in_move,
            jnp.int32(1),
        ),
        n_gen=n_gen,
    )

    assert original.keys() == staged.keys()
    for name in original:
        np.testing.assert_array_equal(staged[name], original[name], err_msg=name)


def test_tiny_high_level_solver_keeps_original_result_contract(tmp_path):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    exact_config = ArtgorExactConfig(
        prefix_bm=2,
        head_bm=2,
        head_bk=2,
        head_bn=2,
        inference_chunk=2,
        parent_chunk=4,
    )
    inference = ArtgorExactInference(
        prefix=lambda states, _weights: states,
        head=lambda hidden, _weights: _oracle_q(
            None, hidden[0], dtype=jnp.bfloat16
        )[None],
    )
    runtime = ArtgorExactBeamRuntime(
        inference=inference,
        weights=None,
        hidden_size=4,
    )
    tree_path = tmp_path / "tiny-tree.u32"
    result = beam_solve_v_only_spmd_packed_exact(
        [0, 1, 2, 3],
        {},
        jnp.asarray([[1, 0, 2, 3], [0, 2, 1, 3]], dtype=jnp.int32),
        jnp.asarray([3, 2, 1, 0], dtype=jnp.uint8),
        jnp.asarray([17, 31, 47, 73], dtype=jnp.int64),
        mesh,
        B_local=8,
        K_per_peer=8,
        n_gen=2,
        state_size=4,
        num_steps=2,
        internal_bs=2,
        tree_path=str(tree_path),
        parent_chunk=4,
        history_depth=1,
        exact_config=exact_config,
        exact_runtime=runtime,
        _require_published_geometry=False,
    )

    required = {
        "found",
        "path_len",
        "path_idx",
        "found_step",
        "wall_s",
        "first_iter_s",
        "last_completed_step",
        "lower_s",
        "compile_s",
        "min_v_trajectory_rank0",
        "timing_breakdown",
    }
    assert required <= result.keys()
    assert not tree_path.exists()
