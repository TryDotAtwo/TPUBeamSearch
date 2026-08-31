"""Behavior contracts for the diagnostic builder and observed intermediates."""
import importlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_layernorm_followup import model_fixture
from tpu_beam_search.artgor_reference import artgor_reference_apply
from tpu_beam_search.stream1_layernorm_reference import layer_norm_reference


def module():
    return importlib.import_module("benchmarks.execution_boundary_ops")


def test_full_control_preserves_model_and_runtime_fp32_embedding():
    params, states, arch, weights = model_fixture()
    weights = weights._replace(embedding=params["embed"])
    call = jax.jit(module().candidate_full(dict(dense="jax", boundary="none"), arch, interpret=True))
    expected = jax.jit(lambda s: artgor_reference_apply(params, s, dtype=jnp.bfloat16))(states)
    np.testing.assert_array_equal(call(states, weights), expected)
    other = weights._replace(output=weights.output._replace(bias=weights.output.bias + 4))
    assert not np.array_equal(call(states, other), expected)


@pytest.mark.parametrize("boundary,count", [("none", 0), ("pre", 1), ("post", 1), ("both", 2)])
def test_barrier_modes_are_threaded_through_dense(boundary, count):
    x, w, b = jnp.ones((2, 8), jnp.bfloat16), jnp.eye(8, dtype=jnp.bfloat16), jnp.zeros(8, jnp.bfloat16)
    call = lambda a, c, d: module().candidate_dense(a, c, d, dict(dense="jax", boundary=boundary))
    closed = jax.make_jaxpr(call)(x, w, b)
    assert sum(e.primitive.name == "optimization_barrier" for e in closed.jaxpr.eqns) == count
    np.testing.assert_array_equal(jax.jit(call)(x, w, b), x)


def test_witness_coordinates_and_nonfinite_are_strict_json():
    got = module().mismatch_witnesses(np.array([[1., 2., 4.]], np.float32),
                                    np.array([[1., 3., np.nan]], np.float32), limit=1)
    assert got["mismatch_count"] == 2
    assert got["examples"][0]["index"] == [0, 1]
    assert got["examples"][0]["reference"] == 2.
    assert got["examples"][0]["candidate"] == 3.
    assert got["shape"] == [1, 3]
    json.dumps(got, allow_nan=False)
    with pytest.raises(ValueError):
        module().mismatch_witnesses(np.zeros(2), np.zeros(3))


def test_ln_instrumentation_keeps_same_source_output_and_statistics_shapes():
    _, _, _, weights = model_fixture()
    x = jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 8)
    norm = weights.residuals[0].first.normalization
    got = jax.jit(module().jax_ln_observe)(x, norm, 1e-5)
    expected = jax.jit(lambda v: layer_norm_reference(v, norm, epsilon=1e-5))(x)
    np.testing.assert_array_equal(got["output"], expected)
    assert got["mean"].shape == (2, 1)
    assert got["variance"].shape == (2, 1)


def test_config_matrix_contains_independent_axes_and_no_unknown_fallback():
    configs = module().dense_configs()
    pallas = [c for c in configs if c["dense"] == "late"]
    assert {(c["bm"], c["bk"], c["bn"]) for c in pallas} == {
        (128, 256, 512), (256, 256, 512), (512, 256, 512), (128, 256, 1024), (128, 1024, 512)}
    with pytest.raises(ValueError):
        module().candidate_dense(jnp.ones((2, 8)), jnp.ones((8, 8)), jnp.ones(8),
                                 dict(dense="jax", boundary="typo"))


def test_observed_node_summaries_do_not_poison_json_with_nan():
    got = module().node_summaries({"value": jnp.array([1., jnp.nan, jnp.inf])})
    assert got["value"]["sample"] == [1., None, None]
    assert got["value"]["finite"] is False
    json.dumps(got, allow_nan=False)


def test_every_full_embedding_variant_uses_runtime_table():
    params, states, arch, weights = model_fixture()
    for encoding in ("jax_flat", "jax_tiled", "pallas_banked"):
        call = jax.jit(module().candidate_full(dict(dense="jax", embedding=encoding, bm=8), arch, interpret=True))
        first = call(states, weights._replace(embedding=params["embed"]))
        changed_table = params["embed"].at[2, 0].add(4)
        second = call(states, weights._replace(embedding=changed_table))
        assert not np.array_equal(first, second)


@pytest.mark.parametrize("storage_dtype", [jnp.bfloat16, jnp.float32])
def test_prepacked_full_candidate_keeps_the_entire_resmlp_exact(storage_dtype):
    params, states, arch, weights = model_fixture()
    embedding = importlib.import_module("tpu_beam_search.stream1_embedding_experimental")
    banks = embedding.prepare_banked_embedding(params["embed"], storage_dtype=storage_dtype)
    packed_weights = weights._replace(embedding=banks)
    config = dict(dense="jax", norm="jax", embedding="pallas_banked_prepacked", bm=8)
    actual = jax.jit(module().candidate_full(config, arch, interpret=True))(states, packed_weights)
    expected = jax.jit(lambda s: artgor_reference_apply(params, s, dtype=jnp.bfloat16))(states)
    np.testing.assert_array_equal(actual, expected)


def test_late_dense_full_builder_preserves_existing_arithmetic_path():
    from benchmarks.stream1_layernorm_followup import full_call
    _, states, arch, weights = model_fixture()
    config = dict(dense="late", norm="jax", bm=2, bk=8, bn=8)
    actual = jax.jit(module().candidate_full(config, arch, interpret=True))(states, weights)
    expected = jax.jit(full_call(config, arch, interpret=True))(states, weights)
    np.testing.assert_array_equal(actual, expected)


def test_provenance_requires_all_four_identical_artifact_domains():
    expected = dict(checkpoint_sha256="checkpoint", original_source_sha256="model",
                    puzzle_sha256="puzzle", input_sha256={"legal": "a", "stress": "b"})
    module().validate_provenance(dict(expected, source_commit="new"), expected)
    for key in expected:
        with pytest.raises(ValueError, match=key):
            module().validate_provenance({**expected, key: "changed"}, expected)
        with pytest.raises(ValueError, match=key):
            module().validate_provenance({k: v for k, v in expected.items() if k != key}, expected)
