from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import jax
import jax.numpy as jnp

from .stream1_architecture import (
    DenseWeights,
    InputEncodingKind,
    LayerNormDenseWeights,
    LayerNormResidualWeights,
    LayerNormStream1Weights,
    LayerNormWeights,
    NormalizationKind,
    Stream1Architecture,
)


def _array(value, dtype):
    return jnp.asarray(value, dtype=dtype)


def _dense(params: Mapping[str, object], prefix: str, *, dtype) -> DenseWeights:
    return DenseWeights(
        weight=_array(params[f"{prefix}_w"], dtype),
        bias=_array(params[f"{prefix}_b"], dtype),
    )


def _normalization(
    params: Mapping[str, object], prefix: str, *, dtype
) -> LayerNormWeights:
    return LayerNormWeights(
        scale=_array(params[f"{prefix}_gamma"], dtype),
        bias=_array(params[f"{prefix}_beta"], dtype),
    )


def layernorm_stream1_weights_from_artgor_params(
    params: Mapping[str, object],
    architecture: Stream1Architecture,
    *,
    dtype=jnp.bfloat16,
) -> LayerNormStream1Weights:
    """Convert Artgor's already-transposed JAX params into typed weights."""

    derived = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=architecture.STATE_STORAGE_LEN
    )
    comparable = replace(architecture, INPUT_ENCODING=derived.INPUT_ENCODING)
    if comparable != derived:
        raise ValueError("Artgor parameter shapes do not match the architecture")

    input_params = params["input_stack"][0]
    residuals = []
    for block in params["res_blocks"]:
        residuals.append(
            LayerNormResidualWeights(
                first=LayerNormDenseWeights(
                    dense=_dense(block, "lin1", dtype=dtype),
                    normalization=_normalization(block, "ln1", dtype=dtype),
                ),
                second=LayerNormDenseWeights(
                    dense=_dense(block, "lin2", dtype=dtype),
                    normalization=_normalization(block, "ln2", dtype=dtype),
                ),
            )
        )
    return LayerNormStream1Weights(
        embedding=_array(params["embed"], dtype),
        input=LayerNormDenseWeights(
            dense=_dense(input_params, "lin", dtype=dtype),
            normalization=_normalization(input_params, "ln", dtype=dtype),
        ),
        residuals=tuple(residuals),
        output=DenseWeights(
            weight=_array(params["head_w"], dtype),
            bias=_array(params["head_b"], dtype),
        ),
    )


def _validate_layernorm_shapes(
    states: jax.Array,
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
) -> None:
    a = architecture
    if a.NORMALIZATION is not NormalizationKind.LAYER_NORM:
        raise ValueError("LayerNorm weights require a LAYER_NORM architecture")
    if states.ndim != 2 or states.shape[1] != a.STATE_STORAGE_LEN:
        raise ValueError(f"states shape must be [batch, {a.STATE_STORAGE_LEN}]")
    expected = (
        ("embedding", weights.embedding.shape, (a.NUM_CLASSES, a.EMBED_DIM)),
        (
            "input weight",
            weights.input.dense.weight.shape,
            (a.STATE_LEN * a.EMBED_DIM, a.HIDDEN1),
        ),
        ("input bias", weights.input.dense.bias.shape, (a.HIDDEN1,)),
        ("input LayerNorm scale", weights.input.normalization.scale.shape, (a.HIDDEN1,)),
        ("input LayerNorm bias", weights.input.normalization.bias.shape, (a.HIDDEN1,)),
        ("output weight", weights.output.weight.shape, (a.HIDDEN2, a.MOVE_COUNT)),
        ("output bias", weights.output.bias.shape, (a.MOVE_COUNT,)),
    )
    for name, actual, wanted in expected:
        if actual != wanted:
            raise ValueError(f"{name} shape must be {wanted}, got {actual}")
    if a.HIDDEN1 != a.HIDDEN2:
        raise ValueError("LayerNorm ResMLP requires HIDDEN1 == HIDDEN2")
    if len(weights.residuals) != a.RESIDUAL_COUNT:
        raise ValueError("residual weights count must equal RESIDUAL_COUNT")
    for index, block in enumerate(weights.residuals):
        for layer_name, layer in (("first", block.first), ("second", block.second)):
            for field_name, actual, wanted in (
                ("weight", layer.dense.weight.shape, (a.HIDDEN2, a.HIDDEN2)),
                ("bias", layer.dense.bias.shape, (a.HIDDEN2,)),
                ("LayerNorm scale", layer.normalization.scale.shape, (a.HIDDEN2,)),
                ("LayerNorm bias", layer.normalization.bias.shape, (a.HIDDEN2,)),
            ):
                if actual != wanted:
                    raise ValueError(
                        f"residual {index} {layer_name} {field_name} shape "
                        f"must be {wanted}, got {actual}"
                    )


def layer_norm_reference(
    values: jax.Array,
    normalization: LayerNormWeights,
    *,
    epsilon: float,
) -> jax.Array:
    """Artgor-compatible LayerNorm; preserves the input computation dtype."""

    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (
        (values - mean)
        * jax.lax.rsqrt(variance + epsilon)
        * normalization.scale
        + normalization.bias
    )


def _reference_input_encoding(
    states: jax.Array,
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
) -> jax.Array:
    # All modes share this semantic oracle. Production Pallas kernels implement
    # different physical mappings and are compared against this result.
    if architecture.INPUT_ENCODING not in tuple(InputEncodingKind):
        raise ValueError("unsupported input encoding")
    embedded = weights.embedding[states.astype(jnp.int32)]
    return embedded.reshape(states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM)


def _normalized_dense(
    values: jax.Array,
    layer: LayerNormDenseWeights,
    *,
    epsilon: float,
    relu: bool,
) -> jax.Array:
    result = values @ layer.dense.weight + layer.dense.bias
    result = layer_norm_reference(
        result, layer.normalization, epsilon=epsilon
    )
    return jax.nn.relu(result) if relu else result


def stream1_layernorm_reference_inference(
    states: jax.Array,
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
) -> jax.Array:
    """Complete LayerNorm ResMLP semantic reference in Artgor operation order."""

    _validate_layernorm_shapes(states, weights, architecture)
    logical_states = states[:, : architecture.STATE_LEN]
    hidden = _reference_input_encoding(logical_states, weights, architecture)
    hidden = _normalized_dense(
        hidden,
        weights.input,
        epsilon=architecture.LAYER_NORM_EPSILON,
        relu=True,
    )
    for block in weights.residuals:
        skip = hidden
        branch = _normalized_dense(
            hidden,
            block.first,
            epsilon=architecture.LAYER_NORM_EPSILON,
            relu=True,
        )
        branch = _normalized_dense(
            branch,
            block.second,
            epsilon=architecture.LAYER_NORM_EPSILON,
            relu=False,
        )
        hidden = jax.nn.relu(skip + branch)
    return hidden @ weights.output.weight + weights.output.bias
