import numpy as np

from benchmarks.artgor_layernorm_invstd import RESULT_NAME, tensor_metrics
from tpu_beam_search.stream1_layernorm_invstd import (
    pallas_invstd,
    pallas_invstd_affine,
)


def test_invstd_result_name_is_stable_for_kaggle_artifacts():
    assert RESULT_NAME == "artgor_layernorm_invstd.json"


def test_interpreted_invstd_exposes_fp32_and_bf16_boundaries():
    import jax
    import jax.numpy as jnp

    variance = jnp.asarray([[0.5], [2.0]], dtype=jnp.bfloat16)
    epsilon = jnp.asarray(1e-5, jnp.bfloat16).astype(jnp.float32)
    expected_fp32 = jax.lax.rsqrt(variance.astype(jnp.float32) + epsilon)
    expected_bf16 = expected_fp32.astype(jnp.bfloat16)
    actual_fp32 = pallas_invstd(
        variance, output_bf16=False, bm=2, interpret=True,
    )
    actual_bf16 = pallas_invstd(
        variance, output_bf16=True, bm=2, interpret=True,
    )
    assert actual_fp32.dtype == jnp.float32
    assert actual_bf16.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(actual_fp32), np.asarray(expected_fp32))
    np.testing.assert_array_equal(np.asarray(actual_bf16), np.asarray(expected_bf16))


def test_interpreted_invstd_affine_matches_explicit_expression():
    import jax.numpy as jnp

    centered = jnp.arange(256, dtype=jnp.float32).reshape(2, 128) / 128 - 1
    invstd = jnp.asarray([[2.0], [0.5]], dtype=jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.25, 0.25, 128).astype(jnp.bfloat16)
    expected = (
        centered * invstd.astype(jnp.float32)
        * scale.astype(jnp.float32)[None, :]
        + bias.astype(jnp.float32)[None, :]
    ).astype(jnp.bfloat16)
    actual = pallas_invstd_affine(
        centered, invstd, scale, bias, bm=2, interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_invstd_metrics_keep_hash_rmse_and_witness():
    expected = np.asarray([1, 2], dtype=np.float32)
    actual = np.asarray([1, 2.5], dtype=np.float32)
    result = tensor_metrics(expected, actual)
    assert result["mismatch_count"] == 1
    assert result["rmse"] == np.sqrt(0.125)
    assert len(result["reference_sha256"]) == 64
    assert result["witnesses"][0]["flat_index"] == 1
