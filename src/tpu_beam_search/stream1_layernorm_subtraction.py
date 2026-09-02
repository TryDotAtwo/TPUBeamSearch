"""Minimal Pallas probes for the centered-subtraction attribution boundary."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tpu_layout import pad_to_multiple


def _matrix_index(row_block):
    return row_block.astype(jnp.int32), jnp.asarray(0, jnp.int32)


def _centered_kernel(values_ref, mean_ref, output_ref):
    output_ref[...] = (
        values_ref[...].astype(jnp.float32)
        - mean_ref[...].astype(jnp.float32)
    )


def _variance_kernel(values_ref, mean_ref, output_ref, *, logical_width: int):
    centered = (
        values_ref[...].astype(jnp.float32)
        - mean_ref[...].astype(jnp.float32)
    )
    variance = (
        jnp.sum(centered * centered, axis=1, keepdims=True) / logical_width
    ).astype(jnp.bfloat16)
    output_ref[...] = jnp.broadcast_to(variance, values_ref.shape)


def _pallas_binary_probe(
    values,
    mean,
    *,
    variance: bool,
    bm: int,
    interpret: bool,
):
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("values must be a nonempty matrix")
    rows, width = values.shape
    if mean.shape != (rows, 1):
        raise ValueError("mean must have shape [rows, 1]")
    if width % 128:
        raise ValueError("subtraction attribution requires width aligned to 128")
    if not isinstance(bm, int) or bm <= 0:
        raise ValueError("bm must be a positive integer")
    padded_rows = pad_to_multiple(rows, bm)
    values_padded = jnp.pad(
        values.astype(jnp.bfloat16), ((0, padded_rows - rows), (0, 0))
    )
    mean_padded = jnp.pad(
        mean.astype(jnp.bfloat16), ((0, padded_rows - rows), (0, 0))
    )
    mean_matrix = jnp.broadcast_to(mean_padded, values_padded.shape)
    spec = pl.BlockSpec((bm, width), _matrix_index)
    kernel = (
        functools.partial(_variance_kernel, logical_width=width)
        if variance else _centered_kernel
    )
    dtype = jnp.bfloat16 if variance else jnp.float32
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[spec, spec],
            out_specs=spec,
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_rows, width), dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=(
            "stream1_centered_variance_probe"
            if variance else "stream1_centered_subtraction_probe"
        ),
    )
    return call(values_padded, mean_matrix)[:rows]


def pallas_centered_subtraction(values, mean, *, bm=128, interpret=False):
    return _pallas_binary_probe(
        values, mean, variance=False, bm=bm, interpret=interpret,
    )


def pallas_centered_variance(values, mean, *, bm=128, interpret=False):
    return _pallas_binary_probe(
        values, mean, variance=True, bm=bm, interpret=interpret,
    )
