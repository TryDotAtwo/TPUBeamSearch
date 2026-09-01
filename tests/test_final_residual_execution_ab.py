"""Contracts for the final-residual schedule-restoration experiment."""
import importlib

import jax
import numpy as np

from benchmarks.execution_boundary_ops import candidate_full
from test_layernorm_followup import model_fixture


def ops():
    return importlib.import_module("benchmarks.final_residual_ops")


def bench():
    return importlib.import_module("benchmarks.stream1_final_residual_ab")


def base_config(**updates):
    config = dict(
        embedding="reference", dense="jax", norm="jax", boundary="none",
        input_boundary="none", final_barrier="none", bm=8,
    )
    config.update(updates)
    return config


def test_every_final_cut_recomposes_the_unmodified_reference():
    _, states, architecture, weights = model_fixture()
    expected = candidate_full(base_config(), architecture)(states, weights)
    for cut in ops().FINAL_CUTS:
        prefix, suffix = ops().candidate_final_partition(
            base_config(), architecture, cut=cut,
        )
        intermediate = prefix(states, weights)
        actual = suffix(intermediate, weights)
        np.testing.assert_array_equal(actual, expected, err_msg=cut)


def test_taps_preserve_q_and_expose_only_one_requested_boundary():
    _, states, architecture, weights = model_fixture()
    expected = candidate_full(base_config(), architecture)(states, weights)
    for tap in ops().FINAL_TAPS:
        q, observed = ops().candidate_final_full(
            base_config(), architecture, tap=tap,
        )(states, weights)
        np.testing.assert_array_equal(q, expected, err_msg=tap)
        assert observed.shape == (len(states), architecture.HIDDEN1)


def test_final_dense2_barrier_is_present_only_when_requested():
    _, states, architecture, weights = model_fixture()
    none = str(jax.make_jaxpr(ops().candidate_final_full(
        base_config(), architecture,
    ))(states, weights))
    targeted = str(jax.make_jaxpr(ops().candidate_final_full(
        base_config(final_barrier="before_and_after_final_dense2"), architecture,
    ))(states, weights))
    assert "optimization_barrier" not in none
    assert targeted.count("optimization_barrier") == 2


def test_target_matrix_pairs_each_tap_and_split_with_jax_control():
    configs = bench().final_residual_configs()
    assert configs[0]["id"] == "original_shard_map"
    assert configs[0]["role"] == "jax_control"
    assert len({config["id"] for config in configs}) == len(configs)
    ids = {config["id"] for config in configs}
    assert {"typed_monolithic", "pallas_monolithic"} <= ids
    for tap in ops().FINAL_TAPS:
        assert {f"typed_tap_{tap}", f"pallas_tap_{tap}"} <= ids
    for cut in ops().FINAL_CUTS:
        assert {f"typed_split_{cut}", f"pallas_split_{cut}"} <= ids
    assert {
        config["final_barrier"] for config in configs
        if config.get("backend") == "barrier"
    } == set(ops().FINAL_BARRIERS) - {"none"}


def test_output_specs_follow_backend_result_tree():
    from jax.sharding import PartitionSpec as P

    assert bench().output_specs_for_config(dict(backend="monolithic")) == P("core", None)
    assert bench().output_specs_for_config(dict(backend="tap")) == (
        P("core", None), P("core", None),
    )
    assert bench().output_specs_for_config(
        dict(backend="split", cut="before_final_block"), stage="prefix",
    ) == P("core", None)
    assert bench().output_specs_for_config(
        dict(backend="split", cut="before_final_dense2"), stage="prefix",
    ) == (P("core", None), P("core", None))


def test_monolithic_barrier_tap_and_split_cases_build_on_cpu(tmp_path):
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from tpu_beam_search.stream1_embedding_experimental import prepare_banked_embedding

    _, states, architecture, weights = model_fixture()
    banks = prepare_banked_embedding(weights.embedding, storage_dtype=jnp.float32)
    banked = weights._replace(embedding=banks)
    prepared = {
        "typed": (candidate_full(base_config(), architecture), weights),
        "pallas": (None, banked),
    }
    selected_ids = {
        "typed_monolithic",
        "pallas_barrier_before_and_after_final_dense2",
        "pallas_tap_after_final_dense2",
        "pallas_split_before_final_dense2",
        "pallas_split_after_final_block",
    }
    configs = [
        {**config, "bm": 8} for config in bench().final_residual_configs()
        if config["id"] in selected_ids
    ]
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device]), ("core",))
    sharding = NamedSharding(mesh, P("core", None))
    sharded_states = jax.device_put(np.asarray(states), sharding)
    replicated_cache = {}
    for config in configs:
        case = bench()._build_case(
            config, prepared=prepared, architecture=architecture,
            states=sharded_states, mesh=mesh,
            replicated_cache=replicated_cache, directory=tmp_path,
            prefix=config["id"], interpret=True,
        )
        initial = case["host_output"](case["initial_output"])
        repeated = case["host_output"](jax.block_until_ready(case["runner"]()))
        np.testing.assert_array_equal(repeated, initial, err_msg=config["id"])
        assert initial.shape == (len(states), architecture.MOVE_COUNT)
        assert np.isfinite(initial.astype(np.float32)).all()
