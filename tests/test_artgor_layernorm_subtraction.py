import numpy as np

from benchmarks.artgor_layernorm_subtraction import (
    CASE_DEFINITIONS,
    RESULT_NAME,
    tensor_metrics,
)
from tpu_beam_search.stream1_layernorm_subtraction import (
    pallas_centered_subtraction,
    pallas_centered_variance,
)


def test_subtraction_protocol_uses_frozen_six_corpora():
    assert RESULT_NAME == "artgor_layernorm_subtraction.json"
    assert tuple(name for name, _, _ in CASE_DEFINITIONS) == (
        "legal_seed_42",
        "legal_seed_142",
        "legal_seed_242",
        "stress_seed_43",
        "stress_seed_143",
        "stress_seed_243",
    )


def test_interpreted_centered_subtraction_matches_explicit_jax():
    import jax.numpy as jnp

    values = jnp.arange(128, dtype=jnp.float32)[None, :].astype(jnp.bfloat16)
    mean = jnp.asarray([[63.5]], dtype=jnp.bfloat16)
    actual = pallas_centered_subtraction(
        values, mean, bm=1, interpret=True,
    )
    expected = values.astype(jnp.float32) - mean.astype(jnp.float32)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_interpreted_fused_variance_matches_explicit_jax():
    import jax.numpy as jnp

    values = jnp.arange(128, dtype=jnp.float32)[None, :].astype(jnp.bfloat16)
    mean = jnp.asarray([[63.5]], dtype=jnp.bfloat16)
    actual = pallas_centered_variance(
        values, mean, bm=1, interpret=True,
    )
    centered = values.astype(jnp.float32) - mean.astype(jnp.float32)
    expected = jnp.mean(centered * centered, axis=1, keepdims=True).astype(
        jnp.bfloat16
    )
    expected = jnp.broadcast_to(expected, values.shape)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_tensor_metrics_include_hash_rmse_and_deterministic_witnesses():
    expected = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    actual = expected.copy()
    actual[0, 1] += 0.5
    result = tensor_metrics(expected, actual, witness_limit=2)
    assert result["mismatch_count"] == 1
    assert result["rmse"] == 0.25
    assert len(result["reference_sha256"]) == 64
    assert len(result["candidate_sha256"]) == 64
    assert result["witnesses"] == [
        {"flat_index": 1, "reference": 2.0, "candidate": 2.5}
    ]
