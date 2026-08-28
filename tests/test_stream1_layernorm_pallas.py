from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.stream1_layernorm_pallas import pallas_layer_norm


def _reference(values, scale, bias, epsilon):
    values_fp32 = values.astype(jnp.float32)
    mean = jnp.mean(values_fp32, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values_fp32 - mean), axis=-1, keepdims=True)
    return (
        (values_fp32 - mean)
        * jax_rsqrt(variance + epsilon)
        * scale.astype(jnp.float32)
        + bias.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def jax_rsqrt(values):
    import jax

    return jax.lax.rsqrt(values)


@pytest.mark.parametrize("width", [8, 130])
def test_pallas_layer_norm_matches_fp32_reference_with_alignment(width):
    values = (
        jnp.arange(3 * width, dtype=jnp.float32).reshape(3, width) / 17
    ).astype(jnp.bfloat16)
    scale = jnp.linspace(0.7, 1.3, width, dtype=jnp.float32).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.2, 0.2, width, dtype=jnp.float32).astype(jnp.bfloat16)
    expected = _reference(values, scale, bias, 1e-5)
    actual = pallas_layer_norm(
        values,
        scale,
        bias,
        bm=4,
        width_alignment=128,
        epsilon=1e-5,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_pallas_layer_norm_constant_rows_are_finite_and_equal_bias():
    values = jnp.full((2, 128), 1000, dtype=jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-1, 1, 128).astype(jnp.bfloat16)
    actual = pallas_layer_norm(
        values, scale, bias, bm=2, epsilon=1e-5, interpret=True
    )
    assert bool(jnp.all(jnp.isfinite(actual)))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(bias[None, :]).repeat(2, 0))


def test_pallas_layer_norm_rejects_non_vector_affine_weights():
    values = jnp.ones((2, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="scale and bias"):
        pallas_layer_norm(
            values,
            jnp.ones((1, 128), dtype=jnp.bfloat16),
            jnp.zeros((128,), dtype=jnp.bfloat16),
            interpret=True,
        )
