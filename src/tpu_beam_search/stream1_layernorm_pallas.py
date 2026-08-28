from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple


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
