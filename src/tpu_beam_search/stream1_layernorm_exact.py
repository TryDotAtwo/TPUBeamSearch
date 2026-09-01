"""Exact high-throughput LayerNorm ResMLP inference for TPU.

The validated TPU-v5-lite path deliberately uses two compiled dispatches.  The
first dispatch performs the prepacked Pallas embedding lookup and the complete
JAX ResMLP body.  The second dispatch applies the output head to the
device-resident hidden matrix.  Keeping this as a real dispatch boundary is
part of the numerical contract: wrapping the composed call in another
``jax.jit`` may merge the stages and recreate the rejected Dense schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .sharding import make_sharded_inference
from .stream1_architecture import (
    DenseWeights,
    LayerNormDenseWeights,
    LayerNormResidualWeights,
    LayerNormStream1Weights,
    NormalizationKind,
    Stream1Architecture,
)
from .stream1_embedding_experimental import (
    BankedEmbedding,
    flat_embedding_prepacked,
    prepare_banked_embedding,
)
from .stream1_layernorm_reference import (
    _validate_layernorm_shapes,
    layer_norm_reference,
)


class ExactLayerNormInferenceWeights(NamedTuple):
    """BF16 model weights plus FP32-stored, BF16-valued embedding banks."""

    embedding: BankedEmbedding
    input: LayerNormDenseWeights
    residuals: tuple[LayerNormResidualWeights, ...]
    output: DenseWeights
    fused_input_weight: jax.Array | None = None


def _require_bf16_model(weights: LayerNormStream1Weights) -> None:
    for leaf in jax.tree.leaves(weights):
        if hasattr(leaf, "dtype") and leaf.dtype != jnp.dtype(jnp.bfloat16):
            raise ValueError("exact LayerNorm inference requires BF16 logical weights")


def prepare_exact_layernorm_inference_weights(
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
) -> ExactLayerNormInferenceWeights:
    """Prepack a checkpoint once for the measured exact TPU inference path."""

    _validate_layernorm_shapes(
        jax.ShapeDtypeStruct(
            (1, architecture.STATE_STORAGE_LEN), jnp.uint8,
        ),
        weights,
        architecture,
    )
    _require_bf16_model(weights)
    return ExactLayerNormInferenceWeights(
        embedding=prepare_banked_embedding(
            weights.embedding, storage_dtype=jnp.float32,
        ),
        input=weights.input,
        residuals=weights.residuals,
        output=weights.output,
        fused_input_weight=weights.fused_input_weight,
    )


def _validate_prepared_shapes(
    states: jax.Array,
    weights: ExactLayerNormInferenceWeights,
    architecture: Stream1Architecture,
) -> None:
    if architecture.NORMALIZATION is not NormalizationKind.LAYER_NORM:
        raise ValueError("exact LayerNorm inference requires a LAYER_NORM architecture")
    if states.ndim != 2 or states.shape[1] != architecture.STATE_STORAGE_LEN:
        raise ValueError(
            f"states shape must be [batch, {architecture.STATE_STORAGE_LEN}]"
        )
    if not isinstance(weights, ExactLayerNormInferenceWeights):
        raise TypeError("weights must be prepared exact LayerNorm inference weights")
    phases = architecture.EMBED_DIM // math.gcd(architecture.EMBED_DIM, 128)
    bank_shape = (phases, 128, 128)
    if (
        weights.embedding.low.shape != bank_shape
        or weights.embedding.high.shape != bank_shape
        or weights.embedding.low.dtype != jnp.dtype(jnp.float32)
        or weights.embedding.high.dtype != jnp.dtype(jnp.float32)
    ):
        raise ValueError(
            f"embedding banks must both be FP32 with shape {bank_shape}"
        )
    logical = LayerNormStream1Weights(
        embedding=jax.ShapeDtypeStruct(
            (architecture.NUM_CLASSES, architecture.EMBED_DIM), jnp.bfloat16,
        ),
        input=weights.input,
        residuals=weights.residuals,
        output=weights.output,
        fused_input_weight=weights.fused_input_weight,
    )
    _validate_layernorm_shapes(states, logical, architecture)
    _require_bf16_model(logical)


def _normalized_dense(values, layer, *, epsilon: float, relu: bool):
    result = values @ layer.dense.weight + layer.dense.bias
    result = layer_norm_reference(
        result, layer.normalization, epsilon=epsilon,
    )
    return jax.nn.relu(result) if relu else result


def stream1_layernorm_exact_prefix(
    states: jax.Array,
    weights: ExactLayerNormInferenceWeights,
    architecture: Stream1Architecture,
    *,
    bm: int = 2048,
    interpret: bool = False,
) -> jax.Array:
    """Pallas lookup followed by the unchanged JAX input and residual body."""

    _validate_prepared_shapes(states, weights, architecture)
    logical_states = states[:, : architecture.STATE_LEN]
    encoded = flat_embedding_prepacked(
        logical_states,
        weights.embedding,
        embed_dim=architecture.EMBED_DIM,
        bm=bm,
        interpret=interpret,
    )
    epsilon = architecture.LAYER_NORM_EPSILON
    hidden = _normalized_dense(
        encoded, weights.input, epsilon=epsilon, relu=True,
    )
    for block in weights.residuals:
        skip = hidden
        branch = _normalized_dense(
            hidden, block.first, epsilon=epsilon, relu=True,
        )
        branch = _normalized_dense(
            branch, block.second, epsilon=epsilon, relu=False,
        )
        hidden = jax.nn.relu(skip + branch)
    return hidden


def stream1_layernorm_exact_head(
    hidden: jax.Array,
    weights: ExactLayerNormInferenceWeights,
    architecture: Stream1Architecture,
) -> jax.Array:
    """Apply the output head in the second compiled dispatch."""

    if hidden.ndim != 2 or hidden.shape[1] != architecture.HIDDEN2:
        raise ValueError(f"hidden shape must be [batch, {architecture.HIDDEN2}]")
    if weights.output.weight.shape != (
        architecture.HIDDEN2, architecture.MOVE_COUNT,
    ) or weights.output.bias.shape != (architecture.MOVE_COUNT,):
        raise ValueError("output head shapes do not match the architecture")
    return hidden @ weights.output.weight + weights.output.bias


@dataclass(frozen=True)
class ShardedExactLayerNormInference:
    """Two separately compiled calls; the intermediate never leaves devices."""

    prefix: Callable
    suffix: Callable

    def __call__(self, states, weights):
        hidden = self.prefix(states, weights)
        return self.suffix(hidden, weights)


def make_sharded_exact_layernorm_inference(
    architecture: Stream1Architecture,
    *,
    mesh,
    weights_example: ExactLayerNormInferenceWeights,
    bm: int = 2048,
    interpret: bool = False,
) -> ShardedExactLayerNormInference:
    """Build the measured two-dispatch path for a ``core`` device mesh.

    States must be sharded on their batch dimension and weights replicated.
    Call the returned object directly from Python; do not enclose its composed
    call in an outer ``jax.jit``, because the materialized boundary is required.
    """

    if "core" not in mesh.axis_names:
        raise ValueError("mesh must contain the 'core' axis")
    dummy_states = jax.ShapeDtypeStruct(
        (1, architecture.STATE_STORAGE_LEN), jnp.uint8,
    )
    _validate_prepared_shapes(dummy_states, weights_example, architecture)
    prefix = make_sharded_inference(
        lambda states, weights: stream1_layernorm_exact_prefix(
            states, weights, architecture, bm=bm, interpret=interpret,
        ),
        mesh=mesh,
        weights_example=weights_example,
    )
    suffix = make_sharded_inference(
        lambda hidden, weights: stream1_layernorm_exact_head(
            hidden, weights, architecture,
        ),
        mesh=mesh,
        weights_example=weights_example,
    )
    return ShardedExactLayerNormInference(prefix=prefix, suffix=suffix)
