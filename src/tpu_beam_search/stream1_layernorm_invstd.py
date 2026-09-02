"""Fixed-operand Pallas probes for variance, invstd and affine boundaries."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple


def _matrix_index(row_block):
    return row_block.astype(jnp.int32), jnp.asarray(0, jnp.int32)


def _vector_index(_row_block):
    return (jnp.asarray(0, jnp.int32),)


def _invstd_kernel(
    variance_ref, output_ref, *, epsilon: float, output_bf16: bool,
):
    epsilon_fp32 = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
    result = jax.lax.rsqrt(
        variance_ref[...].astype(jnp.bfloat16).astype(jnp.float32)
        + epsilon_fp32
    )
    if output_bf16:
        result = result.astype(jnp.bfloat16)
    output_ref[...] = result


def pallas_invstd(
    variance,
    *,
    epsilon: float = 1e-5,
    output_bf16: bool = True,
    bm: int = 128,
    interpret: bool = False,
):
    if variance.ndim != 2 or variance.shape[1] != 1 or variance.shape[0] <= 0:
        raise ValueError("variance must have shape [rows, 1]")
    if not isinstance(bm, int) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    rows = variance.shape[0]
    padded_rows = pad_to_multiple(rows, bm)
    padded = jnp.pad(
        variance.astype(jnp.bfloat16), ((0, padded_rows - rows), (0, 0))
    )
    matrix = jnp.broadcast_to(padded, (padded_rows, 128))
    spec = pl.BlockSpec((bm, 128), _matrix_index)
    dtype = jnp.bfloat16 if output_bf16 else jnp.float32
    call = pl.pallas_call(
        functools.partial(
            _invstd_kernel, epsilon=epsilon, output_bf16=output_bf16,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[spec],
            out_specs=spec,
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_rows, 128), dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=("stream1_invstd_bf16" if output_bf16 else "stream1_invstd_fp32"),
    )
    return call(matrix)[:rows, :1]


def _affine_kernel(centered_ref, invstd_ref, scale_ref, bias_ref, output_ref):
    output_ref[...] = (
        centered_ref[...].astype(jnp.float32)
        * invstd_ref[...].astype(jnp.bfloat16).astype(jnp.float32)
        * scale_ref[...].astype(jnp.bfloat16).astype(jnp.float32)[None, :]
        + bias_ref[...].astype(jnp.bfloat16).astype(jnp.float32)[None, :]
    ).astype(jnp.bfloat16)


def pallas_invstd_affine(
    centered,
    invstd,
    scale,
    bias,
    *,
    bm: int = 128,
    interpret: bool = False,
):
    if centered.ndim != 2 or min(centered.shape) <= 0:
        raise ValueError("centered must be a nonempty matrix")
    rows, width = centered.shape
    if width % 128:
        raise ValueError("affine attribution width must be aligned to 128")
    if invstd.shape != (rows, 1):
        raise ValueError("invstd must have shape [rows, 1]")
    if scale.shape != (width,) or bias.shape != (width,):
        raise ValueError("scale and bias must match centered width")
    padded_rows = pad_to_multiple(rows, bm)
    centered_padded = jnp.pad(
        centered.astype(jnp.float32), ((0, padded_rows - rows), (0, 0))
    )
    invstd_padded = jnp.pad(
        invstd.astype(jnp.bfloat16), ((0, padded_rows - rows), (0, 0))
    )
    invstd_matrix = jnp.broadcast_to(invstd_padded, centered_padded.shape)
    matrix_spec = pl.BlockSpec((bm, width), _matrix_index)
    vector_spec = pl.BlockSpec((width,), _vector_index)
    call = pl.pallas_call(
        _affine_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[matrix_spec, matrix_spec, vector_spec, vector_spec],
            out_specs=matrix_spec,
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_rows, width), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="stream1_invstd_affine",
    )
    return call(
        centered_padded,
        invstd_matrix,
        scale.astype(jnp.bfloat16),
        bias.astype(jnp.bfloat16),
    )[:rows]
