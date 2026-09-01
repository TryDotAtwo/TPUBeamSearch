"""Contracts for the eight-device execution-boundary localization bundle."""
import importlib

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from test_layernorm_followup import model_fixture


def module():
    return importlib.import_module("benchmarks.stream1_inference_execution_ab")


def result(identifier, corpus, latency_ms, *, exact=True, role="candidate"):
    return dict(
        id=identifier,
        corpus=corpus,
        role=role,
        status="ok",
        exact_oracle_on_sample=exact,
        timing_comparable=True,
        timing=dict(median_ms=latency_ms),
    )


def test_execution_matrix_keeps_canonical_oracle_and_crosses_boundaries_and_backends():
    configs = module().execution_configs()
    assert configs[0]["id"] == "original_shard_map"
    assert configs[0]["role"] == "jax_control"
    assert {c["id"] for c in configs} >= {
        "typed_shard_map",
        "pallas_shard_none",
        "pallas_shard_pre",
        "pallas_shard_post",
        "pallas_shard_both",
        "jax_split",
        "pallas_split",
        "original_direct_jit",
        "pallas_direct_jit",
        "original_pmap",
        "pallas_pmap",
        "original_independent",
        "pallas_independent",
    }
    assert len({c["id"] for c in configs}) == len(configs)
    assert all(c["bm"] == 2048 for c in configs if c["role"] == "candidate")


def test_owner_mapping_is_global_row_to_exact_shard_and_local_row():
    assert module().owner_local_index(29_807, local_batch=16_384, device_count=8) == (1, 13_423)
    assert module().owner_local_index(50_224, local_batch=16_384, device_count=8) == (3, 1_072)
    assert module().owner_local_index(29_369, local_batch=16_384, device_count=8) == (1, 12_985)


def test_winner_must_beat_fastest_exact_jax_control_on_every_corpus():
    rows = []
    for corpus in ("legal", "stress"):
        rows.extend([
            result("original_shard_map", corpus, 12.0, role="jax_control"),
            result("typed_shard_map", corpus, 11.5, role="jax_control"),
            result("candidate_fast", corpus, 8.0),
            result("candidate_average", corpus, 7.0 if corpus == "legal" else 12.0),
            result("candidate_inexact", corpus, 6.0, exact=corpus == "legal"),
        ])
    decision = module().select_execution_winner(rows, corpus_names=("legal", "stress"))
    assert decision["winner_id"] == "candidate_fast"
    assert decision["target_achieved"] is True
    assert decision["jax_baseline_id"] == {"legal": "typed_shard_map", "stress": "typed_shard_map"}
    assert decision["per_corpus_speedup"] == {"legal": 11.5 / 8, "stress": 11.5 / 8}


def test_split_helpers_match_monolithic_reference_on_tiny_cpu_fixture():
    from benchmarks.execution_boundary_ops import candidate_encode, candidate_tail

    _, states, architecture, weights = model_fixture()
    config = dict(embedding="reference", dense="jax", norm="jax", boundary="none")
    encoded = candidate_encode(config, architecture)(states, weights)
    split = candidate_tail(config, architecture)(encoded, weights)
    from benchmarks.execution_boundary_ops import candidate_full
    monolithic = candidate_full(config, architecture)(states, weights)
    np.testing.assert_array_equal(split, monolithic)


def test_input_dense_boundary_is_explicit_in_jaxpr():
    from benchmarks.execution_boundary_ops import candidate_full

    _, states, architecture, weights = model_fixture()
    base = dict(embedding="reference", dense="jax", norm="jax", boundary="none")
    none = str(jax.make_jaxpr(candidate_full({**base, "input_boundary": "none"}, architecture))(
        states, weights
    ))
    both = str(jax.make_jaxpr(candidate_full({**base, "input_boundary": "both"}, architecture))(
        states, weights
    ))
    assert "optimization_barrier" not in none
    assert both.count("optimization_barrier") >= 2


def test_boundary_nodes_include_encoded_input_dense_each_block_and_q():
    from benchmarks.execution_boundary_ops import candidate_nodes

    _, states, architecture, weights = model_fixture()
    config = dict(embedding="reference", dense="jax", norm="jax", boundary="none")
    nodes = candidate_nodes(config, architecture, sample_rows=(0,))(states, weights)
    assert tuple(nodes) == (
        "encoded", "input_dense", "input_hidden",
        *(f"block_{index}" for index in range(architecture.RESIDUAL_COUNT)),
        "q",
    )
    assert nodes["encoded"].shape == (1, architecture.STATE_LEN * architecture.EMBED_DIM)
    assert nodes["input_dense"].shape == (1, architecture.HIDDEN1)
    assert nodes["q"].shape == (1, architecture.MOVE_COUNT)
    assert all(value.dtype == jnp.bfloat16 for value in nodes.values())


def test_all_execution_backends_build_and_repeat_deterministically(tmp_path):
    from tpu_beam_search.stream1_embedding_experimental import prepare_banked_embedding

    _, states, architecture, weights = model_fixture()
    banks = prepare_banked_embedding(weights.embedding, storage_dtype=jnp.float32)
    banked = weights._replace(embedding=banks)
    prepared = {
        "pallas": (None, banked),
        "typed": (lambda x, w: candidate_full(
            dict(embedding="reference", dense="jax", norm="jax", boundary="none"),
            architecture, interpret=True,
        )(x, w), weights),
    }
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device]), ("core",))
    host = np.asarray(states)
    state = jax.device_put(host, NamedSharding(mesh, P("core", None)))
    got = {}
    for backend in ("shard_map", "split", "direct_jit", "pmap", "independent"):
        config = dict(
            id=f"tiny_{backend}", role="candidate", backend=backend,
            implementation="pallas", embedding="pallas_banked_prepacked",
            dense="jax", norm="jax", boundary="none", input_boundary="none", bm=8,
        )
        case = module()._build_case(
            config, prepared=prepared, architecture=architecture,
            states=state, host_states=host, mesh=mesh, devices=[device],
            replicated_cache={}, pmap_cache={}, independent_cache={},
            directory=tmp_path, prefix=config["id"], interpret=True,
        )
        initial = case["host_output"](case["initial_output"])
        repeated = case["host_output"](jax.block_until_ready(case["runner"]()))
        got[backend] = initial
        np.testing.assert_array_equal(repeated, initial, err_msg=backend)
        assert initial.shape == (len(states), architecture.MOVE_COUNT)
        assert np.isfinite(initial.astype(np.float32)).all()
    assert set(got) == {"shard_map", "split", "direct_jit", "pmap", "independent"}
