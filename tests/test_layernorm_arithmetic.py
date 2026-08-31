"""CPU interpreter witnesses, not evidence of TPU accuracy or performance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tpu_beam_search.stream1_layernorm_pallas as kernels
from tpu_beam_search.stream1_architecture import InputEncodingKind, Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference,
    layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


def _random_model(width=8):
    rng = np.random.default_rng(4)

    def array(shape):
        return jnp.asarray(rng.normal(size=shape), jnp.bfloat16)

    def layer(prefix, norm, input_width):
        return {
            f"{prefix}_w": array((input_width, width)),
            f"{prefix}_b": array((width,)),
            f"{norm}_gamma": array((width,)),
            f"{norm}_beta": array((width,)),
        }

    params = {
        "encoding": "embedding", "state_size": 2, "num_classes": 5,
        "d_model": width, "output_dim": 3, "embed": array((5, 4)),
        "input_stack": [layer("lin", "ln", 8)],
        "res_blocks": [dict(layer("lin1", "ln1", width),
                            **layer("lin2", "ln2", width))],
        "head_w": array((width, 3)), "head_b": array((3,)),
    }
    architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=4)
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    states = jnp.asarray([[0, 1, 255, 255], [4, 2, 255, 255]], jnp.uint8)
    return states, weights, architecture


def _bf16_ln(values, scale, bias, epsilon=1e-5):
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


@pytest.mark.parametrize("bm,bk,bn", [(4, 8, 8), (3, 4, 5)])
def test_dense_rounds_dot_before_bias_and_preserves_legacy_difference(bm, bk, bn):
    rng = np.random.default_rng(4)
    x, w, b = [jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
               for shape in ((4, 8), (8, 8), (8,))]
    expected = x @ w + b
    assert hasattr(kernels, "pallas_layernorm_dense")
    early = kernels.pallas_layernorm_dense(
        x, w, b, bm=bm, bk=bk, bn=bn,
        dense_rounding="bf16_before_bias", interpret=True,
    )
    late = kernels.pallas_layernorm_dense(x, w, b, bm=bm, bk=bk, bn=bn, interpret=True)
    np.testing.assert_array_equal(early, expected)
    assert np.any(np.asarray(late) != np.asarray(expected))


def test_bf16_mean_rounds_after_division_over_logical_not_padded_width():
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(16, 130)), jnp.bfloat16)
    scale = jnp.ones((130,), jnp.bfloat16)
    bias = jnp.zeros((130,), jnp.bfloat16)
    expected = _bf16_ln(x, scale, bias)
    matched = kernels.pallas_layer_norm(
        x, scale, bias, bm=4, width_alignment=128,
        fp32_statistics=False, mean_mode="jax", interpret=True,
    )
    legacy = kernels.pallas_layer_norm(
        x, scale, bias, bm=4, width_alignment=128,
        fp32_statistics=False, interpret=True,
    )
    np.testing.assert_array_equal(matched, expected)
    assert np.any(np.asarray(legacy) != np.asarray(expected))


def test_fused_dense_propagates_early_rounding_and_logical_mean():
    rng = np.random.default_rng(7)
    x, w, b = [jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
               for shape in ((2, 8), (8, 130), (130,))]
    scale = jnp.ones((130,), jnp.bfloat16)
    beta = jnp.zeros((130,), jnp.bfloat16)
    expected = jax.nn.relu(_bf16_ln(x @ w + b, scale, beta))
    actual = kernels.pallas_fused_dense_layer_norm(
        x, w, b, scale, beta, bm=2, bk=8, bn=256,
        fp32_statistics=False, dense_rounding="bf16_before_bias", mean_mode="jax",
        relu=True, interpret=True,
    )
    np.testing.assert_array_equal(actual, expected)


def test_fused_block_propagates_arithmetic_through_both_dense_and_norms():
    _, weights, architecture = _random_model(130)
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(2, 130)), jnp.bfloat16)
    block = weights.residuals[0]
    y = x
    for index, layer in enumerate((block.first, block.second)):
        y = layer_norm_reference(y @ layer.dense.weight + layer.dense.bias,
                                 layer.normalization,
                                 epsilon=architecture.LAYER_NORM_EPSILON)
        if index == 0:
            y = jax.nn.relu(y)
    expected = jax.nn.relu(x + y)
    actual = kernels.pallas_fused_residual_block(
        x, block, bm=2, bk=256, bn=256, fp32_statistics=False,
        dense_rounding="bf16_before_bias", mean_mode="jax", interpret=True,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fusion", ["separate", "per_layer", "per_block"])
@pytest.mark.parametrize("width", [8, 130])
def test_full_inference_propagates_arithmetic_including_prefix_and_q_head(fusion, width):
    states, weights, architecture = _random_model(width)
    tile = 8 if width == 8 else 256
    expected = stream1_layernorm_reference_inference(states, weights, architecture)
    actual = kernels.stream1_layernorm_pallas_inference(
        states, weights, architecture, bm=2,
        bk_input=tile, bn_input=tile, bk_hidden=tile, bn_hidden=tile,
        bk_output=tile, bn_output=tile,
        layernorm_fusion=fusion, fp32_statistics=False,
        dense_rounding="bf16_before_bias", mean_mode="jax", interpret=True,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("option,value", [("dense_rounding", "typo"), ("mean_mode", "typo")])
def test_full_inference_rejects_unknown_arithmetic(option, value):
    states, weights, architecture = _random_model()
    with pytest.raises(ValueError, match=option):
        kernels.stream1_layernorm_pallas_inference(
            states, weights, architecture, interpret=True, **{option: value}
        )


@pytest.mark.parametrize("encoding", [InputEncodingKind.VIRTUAL_ONE_HOT_MXU,
                                     InputEncodingKind.FUSED_VIRTUAL_ONE_HOT])
def test_non_embedding_prefix_rejects_unimplemented_early_rounding(encoding):
    states, weights, architecture = _random_model()
    with pytest.raises(ValueError, match="embedding"):
        kernels.pallas_layernorm_input_prefix(
            states, weights, architecture, input_encoding=encoding,
            dense_rounding="bf16_before_bias", interpret=True,
        )
