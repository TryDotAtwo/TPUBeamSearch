from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tpu_beam_search.stream1_layernorm_pallas as layernorm_pallas

from tpu_beam_search.stream1_architecture import InputEncodingKind, Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas import (
    make_fused_virtual_one_hot_weight,
    pallas_layer_norm,
    pallas_layernorm_input_prefix,
    stream1_layernorm_pallas_inference,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
    layer_norm_reference,
)


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


def test_pallas_layer_norm_can_match_artgor_bf16_statistics():
    values = (
        jnp.arange(3 * 128, dtype=jnp.float32).reshape(3, 128) / 17
    ).astype(jnp.bfloat16)
    scale = jnp.linspace(0.7, 1.3, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.2, 0.2, 128).astype(jnp.bfloat16)
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    expected = (
        (values - mean) * jax_rsqrt(variance + 1e-5) * scale + bias
    )

    actual = pallas_layer_norm(
        values,
        scale,
        bias,
        bm=4,
        fp32_statistics=False,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_pallas_layer_norm_rejects_non_vector_affine_weights():
    values = jnp.ones((2, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="scale and bias"):
        pallas_layer_norm(
            values,
            jnp.ones((1, 128), dtype=jnp.bfloat16),
            jnp.zeros((128,), dtype=jnp.bfloat16),
            interpret=True,
        )


def _prefix_fixture():
    hidden = 8
    params = {
        "encoding": "embedding",
        "state_size": 2,
        "num_classes": 5,
        "d_model": hidden,
        "output_dim": 3,
        "embed": jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 11,
        "input_stack": [{
            "lin_w": jnp.arange(48, dtype=jnp.float32).reshape(6, 8) / 37,
            "lin_b": jnp.linspace(-0.2, 0.2, hidden),
            "ln_gamma": jnp.linspace(0.8, 1.2, hidden),
            "ln_beta": jnp.linspace(-0.1, 0.1, hidden),
        }],
        "res_blocks": [{
            "lin1_w": jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 43,
            "lin1_b": jnp.linspace(-0.1, 0.1, hidden),
            "ln1_gamma": jnp.linspace(0.9, 1.1, hidden),
            "ln1_beta": jnp.linspace(-0.05, 0.05, hidden),
            "lin2_w": jnp.flip(
                jnp.arange(64, dtype=jnp.float32).reshape(8, 8), axis=0
            ) / 41,
            "lin2_b": jnp.linspace(0.1, -0.1, hidden),
            "ln2_gamma": jnp.linspace(1.1, 0.9, hidden),
            "ln2_beta": jnp.linspace(0.05, -0.05, hidden),
        }],
        "head_w": jnp.arange(24, dtype=jnp.float32).reshape(8, 3) / 23,
        "head_b": jnp.linspace(-0.2, 0.2, 3),
    }
    architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=4)
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    states = jnp.asarray([[0, 1, 9, 9], [4, 3, 8, 8]], dtype=jnp.uint8)
    return states, weights, architecture


@pytest.mark.parametrize("encoding", list(InputEncodingKind))
def test_pallas_input_candidates_match_fp32_prefix_reference(encoding):
    states, weights, architecture = _prefix_fixture()
    logical = states[:, : architecture.STATE_LEN]
    encoded = weights.embedding[logical.astype(jnp.int32)].reshape(2, -1)
    dense = (
        encoded.astype(jnp.float32)
        @ weights.input.dense.weight.astype(jnp.float32)
        + weights.input.dense.bias.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    expected = jnp.maximum(
        layer_norm_reference(
            dense.astype(jnp.float32),
            weights.input.normalization,
            epsilon=architecture.LAYER_NORM_EPSILON,
        ),
        0,
    ).astype(jnp.bfloat16)
    fused = make_fused_virtual_one_hot_weight(
        weights.embedding,
        weights.input.dense.weight,
        STATE_LEN=architecture.STATE_LEN,
    )
    actual = pallas_layernorm_input_prefix(
        states,
        weights,
        architecture,
        input_encoding=encoding,
        fused_input_weight=fused,
        bm=2,
        bk=8,
        bn=8,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        atol=0.0625,
    )


def test_complete_pallas_layernorm_resmlp_matches_reference():
    from tpu_beam_search.stream1_layernorm_reference import (
        stream1_layernorm_reference_inference,
    )

    states, weights, architecture = _prefix_fixture()
    expected = stream1_layernorm_reference_inference(states, weights, architecture)
    actual = stream1_layernorm_pallas_inference(
        states,
        weights,
        architecture,
        input_encoding=InputEncodingKind.EMBEDDING_GATHER,
        bm=2,
        bk_input=8,
        bn_input=8,
        bk_hidden=8,
        bn_hidden=8,
        bk_output=8,
        bn_output=8,
        interpret=True,
    )
    assert actual.shape == (2, architecture.MOVE_COUNT)
    assert bool(jnp.all(jnp.isfinite(actual)))
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        atol=0.125,
    )


def test_complete_fused_layernorm_resmlp_matches_separate_kernels():
    states, weights, architecture = _prefix_fixture()
    separate = stream1_layernorm_pallas_inference(
        states,
        weights,
        architecture,
        input_encoding=InputEncodingKind.EMBEDDING_GATHER,
        layernorm_fusion="separate",
        bm=2,
        bk_input=8,
        bn_input=8,
        bk_hidden=8,
        bn_hidden=8,
        bk_output=8,
        bn_output=8,
        interpret=True,
    )
    fused = stream1_layernorm_pallas_inference(
        states,
        weights,
        architecture,
        input_encoding=InputEncodingKind.EMBEDDING_GATHER,
        layernorm_fusion="per_layer",
        bm=2,
        bk_input=8,
        bn_input=8,
        bk_hidden=8,
        bn_hidden=8,
        bk_output=8,
        bn_output=8,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(fused), np.asarray(separate))


def test_complete_bf16_statistics_fusion_matches_separate_kernels():
    states, weights, architecture = _prefix_fixture()
    common = dict(
        input_encoding=InputEncodingKind.EMBEDDING_GATHER,
        fp32_statistics=False,
        bm=2,
        bk_input=8,
        bn_input=8,
        bk_hidden=8,
        bn_hidden=8,
        bk_output=8,
        bn_output=8,
        interpret=True,
    )
    separate = stream1_layernorm_pallas_inference(
        states, weights, architecture, layernorm_fusion="separate", **common
    )
    fused = stream1_layernorm_pallas_inference(
        states, weights, architecture, layernorm_fusion="per_layer", **common
    )
    np.testing.assert_array_equal(np.asarray(fused), np.asarray(separate))


@pytest.mark.parametrize("add_skip,relu", [(False, True), (True, True), (False, False)])
def test_fused_dense_layer_norm_matches_fp32_reference(add_skip, relu):
    values = (jnp.arange(24, dtype=jnp.float32).reshape(3, 8) / 13).astype(
        jnp.bfloat16
    )
    weight = (jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 29).astype(
        jnp.bfloat16
    )
    bias = jnp.linspace(-0.2, 0.2, 8).astype(jnp.bfloat16)
    scale = jnp.linspace(0.8, 1.2, 8).astype(jnp.bfloat16)
    beta = jnp.linspace(-0.1, 0.1, 8).astype(jnp.bfloat16)
    skip = jnp.flip(values, axis=1)
    dense = (
        values.astype(jnp.float32) @ weight.astype(jnp.float32)
        + bias.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    expected = _reference(dense, scale, beta, 1e-5)
    if add_skip:
        expected = expected.astype(jnp.float32) + skip.astype(jnp.float32)
    if relu:
        expected = jnp.maximum(expected, 0).astype(jnp.bfloat16)
    else:
        expected = expected.astype(jnp.bfloat16)

    actual = layernorm_pallas.pallas_fused_dense_layer_norm(
        values,
        weight,
        bias,
        scale,
        beta,
        skip=skip,
        add_skip=add_skip,
        relu=relu,
        bm=4,
        bk=8,
        bn=8,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        atol=0,
    )
