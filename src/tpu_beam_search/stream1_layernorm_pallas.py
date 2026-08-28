from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple
from .stream1_architecture import InputEncodingKind, LayerNormStream1Weights, Stream1Architecture
from .stream1_pallas import pallas_dense_linear, pallas_folded_input_linear


def _layer_norm_kernel(
    values_ref,
    scale_ref,
    bias_ref,
    output_ref,
    *,
    logical_width: int,
    epsilon: float,
):
    values = values_ref[...].astype(jnp.float32)
    columns = jnp.arange(values.shape[1], dtype=jnp.int32)
    valid = columns < logical_width
    masked_values = jnp.where(valid[None, :], values, 0.0)
    mean = jnp.sum(masked_values, axis=1, keepdims=True) / logical_width
    centered = jnp.where(valid[None, :], values - mean, 0.0)
    variance = jnp.sum(jnp.square(centered), axis=1, keepdims=True) / logical_width
    normalized = centered * jax.lax.rsqrt(variance + epsilon)
    result = (
        normalized * scale_ref[...][None, :].astype(jnp.float32)
        + bias_ref[...][None, :].astype(jnp.float32)
    )
    output_ref[...] = jnp.where(valid[None, :], result, 0.0).astype(jnp.bfloat16)


def pallas_layer_norm(
    values,
    scale,
    bias,
    *,
    bm: int = 128,
    width_alignment: int = 128,
    epsilon: float = 1e-5,
    interpret: bool = False,
):
    """Per-row LayerNorm with FP32 reductions and aligned BF16 storage."""

    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    rows, logical_width = values.shape
    if scale.shape != (logical_width,) or bias.shape != (logical_width,):
        raise ValueError("scale and bias must be vectors matching values width")
    if bm <= 0 or width_alignment <= 0:
        raise ValueError("bm and width_alignment must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(logical_width, width_alignment)
    values_padded = jnp.pad(
        values.astype(jnp.bfloat16),
        ((0, padded_rows - rows), (0, padded_width - logical_width)),
    )
    scale_padded = jnp.pad(
        scale.astype(jnp.bfloat16), ((0, padded_width - logical_width),)
    )
    bias_padded = jnp.pad(
        bias.astype(jnp.bfloat16), ((0, padded_width - logical_width),)
    )

    call = pl.pallas_call(
        functools.partial(
            _layer_norm_kernel,
            logical_width=logical_width,
            epsilon=epsilon,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (bm, padded_width), lambda row_block: (row_block, 0)
                ),
                pl.BlockSpec((padded_width,), lambda row_block: (0,)),
                pl.BlockSpec((padded_width,), lambda row_block: (0,)),
            ],
            out_specs=pl.BlockSpec(
                (bm, padded_width), lambda row_block: (row_block, 0)
            ),
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_width), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
        interpret=interpret,
        name="stream1_layer_norm",
    )
    return call(values_padded, scale_padded, bias_padded)[:rows, :logical_width]


def make_fused_virtual_one_hot_weight(
    embedding,
    input_weight,
    *,
    STATE_LEN: int,
):
    """Fold embedding and first dense weights outside the timed inference call."""

    num_classes, embed_dim = embedding.shape
    if input_weight.shape[0] != STATE_LEN * embed_dim:
        raise ValueError("input weight rows must equal STATE_LEN * EMBED_DIM")
    hidden = input_weight.shape[1]
    per_position = input_weight.reshape(STATE_LEN, embed_dim, hidden)
    return jnp.einsum(
        "ce,seh->sch",
        embedding.astype(jnp.float32),
        per_position.astype(jnp.float32),
    ).reshape(STATE_LEN * num_classes, hidden).astype(jnp.bfloat16)


def pallas_layernorm_input_prefix(
    states,
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
    *,
    input_encoding: InputEncodingKind,
    fused_input_weight=None,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    interpret: bool = False,
):
    """Execute encoding, first dense, FP32 LayerNorm, and ReLU."""

    logical_states = states[:, : architecture.STATE_LEN]
    if input_encoding is InputEncodingKind.EMBEDDING_GATHER:
        encoded = weights.embedding[logical_states.astype(jnp.int32)].reshape(
            states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM
        ).astype(jnp.bfloat16)
        hidden = pallas_dense_linear(
            encoded,
            weights.input.dense.weight,
            weights.input.dense.bias,
            bm=bm,
            bk=bk,
            bn=bn,
            relu=False,
            interpret=interpret,
        )
    elif input_encoding is InputEncodingKind.VIRTUAL_ONE_HOT_MXU:
        flattened_states = logical_states.reshape(-1, 1)
        zero_embedding_bias = jnp.zeros(
            (architecture.EMBED_DIM,), dtype=jnp.bfloat16
        )
        encoded = pallas_folded_input_linear(
            flattened_states,
            weights.embedding,
            zero_embedding_bias,
            STATE_LEN=1,
            NUM_CLASSES=architecture.NUM_CLASSES,
            bm=bm,
            bk=bk,
            bn=bn,
            relu=False,
            interpret=interpret,
        ).reshape(
            states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM
        )
        hidden = pallas_dense_linear(
            encoded,
            weights.input.dense.weight,
            weights.input.dense.bias,
            bm=bm,
            bk=bk,
            bn=bn,
            relu=False,
            interpret=interpret,
        )
    elif input_encoding is InputEncodingKind.FUSED_VIRTUAL_ONE_HOT:
        if fused_input_weight is None:
            raise ValueError("fused_input_weight is required for fused encoding")
        hidden = pallas_folded_input_linear(
            logical_states,
            fused_input_weight,
            weights.input.dense.bias,
            STATE_LEN=architecture.STATE_LEN,
            NUM_CLASSES=architecture.NUM_CLASSES,
            bm=bm,
            bk=bk,
            bn=bn,
            relu=False,
            interpret=interpret,
        )
    else:
        raise ValueError(f"unsupported input encoding: {input_encoding}")

    normalized = pallas_layer_norm(
        hidden,
        weights.input.normalization.scale,
        weights.input.normalization.bias,
        bm=bm,
        epsilon=architecture.LAYER_NORM_EPSILON,
        interpret=interpret,
    )
    return jax.nn.relu(normalized).astype(jnp.bfloat16)
