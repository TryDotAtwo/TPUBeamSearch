"""Observable one-output Pallas probes for LayerNorm arithmetic attribution."""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple


CHECKPOINT_DTYPES = {
    "mean": jnp.bfloat16,
    "centered": jnp.float32,
    "variance": jnp.bfloat16,
    "invstd": jnp.bfloat16,
    "affine_fp32": jnp.float32,
    "affine_bf16": jnp.bfloat16,
    "relu": jnp.bfloat16,
}


def _matrix_index(row_block):
    return row_block.astype(jnp.int32), jnp.asarray(0, jnp.int32)


def _vector_index(_row_block):
    return (jnp.asarray(0, jnp.int32),)


def _probe_kernel(
    values_ref,
    scale_ref,
    bias_ref,
    output_ref,
    *,
    checkpoint: str,
    logical_width: int,
    epsilon: float,
):
    values_bf16 = values_ref[...].astype(jnp.bfloat16)
    columns = jax.lax.broadcasted_iota(jnp.int32, values_bf16.shape, 1)
    valid = columns < logical_width
    values_fp32 = values_bf16.astype(jnp.float32)
    masked = jnp.where(valid, values_fp32, jnp.float32(0))
    mean = (
        jnp.sum(masked, axis=1, keepdims=True) / logical_width
    ).astype(jnp.bfloat16)
    centered = jnp.where(
        valid, values_fp32 - mean.astype(jnp.float32), jnp.float32(0)
    )
    variance = (
        jnp.sum(centered * centered, axis=1, keepdims=True) / logical_width
    ).astype(jnp.bfloat16)
    epsilon_fp32 = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
    invstd = jax.lax.rsqrt(
        variance.astype(jnp.float32) + epsilon_fp32
    ).astype(jnp.bfloat16)
    affine_fp32 = (
        centered
        * invstd.astype(jnp.float32)
        * scale_ref[...].astype(jnp.float32)[None, :]
        + bias_ref[...].astype(jnp.float32)[None, :]
    )
    affine_bf16 = affine_fp32.astype(jnp.bfloat16)
    relu = jnp.maximum(affine_bf16, jnp.bfloat16(0)).astype(jnp.bfloat16)
    selected = {
        "mean": mean,
        "centered": centered,
        "variance": variance,
        "invstd": invstd,
        "affine_fp32": affine_fp32,
        "affine_bf16": affine_bf16,
        "relu": relu,
    }[checkpoint]
    output_ref[...] = jnp.where(
        valid, selected, jnp.asarray(0, CHECKPOINT_DTYPES[checkpoint])
    )


def pallas_layernorm_probe(
    values,
    scale,
    bias,
    *,
    checkpoint: str,
    epsilon: float = 1e-5,
    bm: int = 128,
    width_alignment: int = 128,
    interpret: bool = False,
):
    """Materialize exactly one LayerNorm boundary as one Pallas dispatch."""

    if checkpoint not in CHECKPOINT_DTYPES:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("values must be a nonempty matrix")
    rows, logical_width = values.shape
    if scale.shape != (logical_width,) or bias.shape != (logical_width,):
        raise ValueError("scale and bias must match values width")
    if not isinstance(bm, int) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    if not isinstance(width_alignment, int) or width_alignment <= 0:
        raise ValueError("width_alignment must be a positive integer")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(logical_width, width_alignment)
    padding = ((0, padded_rows - rows), (0, padded_width - logical_width))
    values_padded = jnp.pad(values.astype(jnp.bfloat16), padding)
    scale_padded = jnp.pad(
        scale.astype(jnp.bfloat16), (0, padded_width - logical_width)
    )
    bias_padded = jnp.pad(
        bias.astype(jnp.bfloat16), (0, padded_width - logical_width)
    )
    matrix_spec = pl.BlockSpec((bm, padded_width), _matrix_index)
    vector_spec = pl.BlockSpec((padded_width,), _vector_index)
    dtype = CHECKPOINT_DTYPES[checkpoint]
    call = pl.pallas_call(
        functools.partial(
            _probe_kernel,
            checkpoint=checkpoint,
            logical_width=logical_width,
            epsilon=epsilon,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[matrix_spec, vector_spec, vector_spec],
            out_specs=matrix_spec,
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_rows, padded_width), dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=f"stream1_ln_probe_{checkpoint}",
    )
    return call(values_padded, scale_padded, bias_padded)[:rows, :logical_width]
