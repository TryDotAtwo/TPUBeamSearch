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


def _variance_from_centered_kernel(
    centered_ref, output_ref, *, logical_width: int,
):
    centered = centered_ref[...].astype(jnp.float32)
    variance = (
        jnp.sum(centered * centered, axis=1, keepdims=True) / logical_width
    ).astype(jnp.bfloat16)
    output_ref[...] = jnp.broadcast_to(variance, centered_ref.shape)


def pallas_variance_from_centered(
    centered, *, bm: int = 128, interpret: bool = False,
):
    if centered.ndim != 2 or min(centered.shape) <= 0:
        raise ValueError("centered must be a nonempty matrix")
    rows, width = centered.shape
    if width % 128:
        raise ValueError("variance attribution width must be aligned to 128")
    padded_rows = pad_to_multiple(rows, bm)
    padded = jnp.pad(
        centered.astype(jnp.float32), ((0, padded_rows - rows), (0, 0))
    )
    spec = pl.BlockSpec((bm, width), _matrix_index)
    call = pl.pallas_call(
        functools.partial(_variance_from_centered_kernel, logical_width=width),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[spec], out_specs=spec,
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_rows, width), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="stream1_variance_from_centered",
    )
    return call(padded)[:rows, :1]


def _affine_kernel(
    centered_ref, invstd_ref, scale_ref, bias_ref, skip_ref, output_ref,
    *, add_skip: bool, relu: bool,
):
    result = (
        centered_ref[...].astype(jnp.float32)
        * invstd_ref[...].astype(jnp.bfloat16).astype(jnp.float32)
        * scale_ref[...].astype(jnp.bfloat16).astype(jnp.float32)[None, :]
        + bias_ref[...].astype(jnp.bfloat16).astype(jnp.float32)[None, :]
    ).astype(jnp.bfloat16)
    if add_skip:
        result = (result + skip_ref[...].astype(jnp.bfloat16)).astype(jnp.bfloat16)
    if relu:
        result = jnp.maximum(result, jnp.bfloat16(0)).astype(jnp.bfloat16)
    output_ref[...] = result


def pallas_invstd_affine(
    centered,
    invstd,
    scale,
    bias,
    *,
    skip=None,
    add_skip: bool = False,
    relu: bool = False,
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
    if add_skip and (skip is None or skip.shape != centered.shape):
        raise ValueError("add_skip requires skip matching centered shape")
    padded_rows = pad_to_multiple(rows, bm)
    centered_padded = jnp.pad(
        centered.astype(jnp.float32), ((0, padded_rows - rows), (0, 0))
    )
    invstd_padded = jnp.pad(
        invstd.astype(jnp.bfloat16), ((0, padded_rows - rows), (0, 0))
    )
    invstd_matrix = jnp.broadcast_to(invstd_padded, centered_padded.shape)
    skip_source = centered if skip is None else skip
    skip_padded = jnp.pad(
        skip_source.astype(jnp.bfloat16), ((0, padded_rows - rows), (0, 0))
    )
    matrix_spec = pl.BlockSpec((bm, width), _matrix_index)
    vector_spec = pl.BlockSpec((width,), _vector_index)
    call = pl.pallas_call(
        functools.partial(_affine_kernel, add_skip=add_skip, relu=relu),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                matrix_spec, matrix_spec, vector_spec, vector_spec, matrix_spec,
            ],
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
        skip_padded,
    )[:rows]
