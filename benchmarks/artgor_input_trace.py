"""Diagnostic only: materialized intermediates are not a monolithic oracle."""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

TRACE_NAMES = ('sum', 'mean_bf16', 'centered', 'variance', 'invstd_bf16', 'output')


def _mean_kernel(raw, out):
    out[...] = jnp.broadcast_to(
        (jnp.sum(raw[...], axis=1, keepdims=True) / raw.shape[1]).astype(jnp.bfloat16), raw.shape)


def mean_buffer(raw, *, pallas=False, interpret=False, bm=128):
    """Return a genuinely BF16 buffer, not a widened round-trip trace."""
    if not pallas:
        return jnp.broadcast_to(
            (jnp.sum(raw, axis=1, keepdims=True) / raw.shape[1]).astype(jnp.bfloat16), raw.shape)
    matrix = pl.BlockSpec((bm, raw.shape[1]), lambda i: (i.astype(jnp.int32), jnp.int32(0)))
    return pl.pallas_call(
        _mean_kernel, out_shape=jax.ShapeDtypeStruct(raw.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[matrix], out_specs=matrix, grid=(raw.shape[0] // bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret, name='raw_mean_bf16_buffer',
    )(raw)


def _external_kernel(raw, mean, scale, bias, out, *, epsilon):
    centered = raw[...].astype(jnp.bfloat16).astype(jnp.float32) - mean[...].astype(jnp.float32)
    variance = jnp.sum(centered * centered, axis=1, keepdims=True) / raw.shape[1]
    invstd = jax.lax.rsqrt(variance + jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32))
    invstd = invstd.astype(jnp.bfloat16).astype(jnp.float32)
    out[...] = jnp.maximum(centered * invstd * scale[...].astype(jnp.float32)[None, :]
                           + bias[...].astype(jnp.float32)[None, :], 0).astype(jnp.bfloat16)


def external_mean_ln(raw, mean, scale, bias, *, epsilon=1e-5, interpret=False, bm=128):
    if mean.shape != raw.shape or mean.dtype != jnp.bfloat16:
        raise ValueError('external mean must be a matching BF16 matrix')
    matrix = pl.BlockSpec((bm, raw.shape[1]), lambda i: (i.astype(jnp.int32), jnp.int32(0)))
    vector = pl.BlockSpec((raw.shape[1],), lambda i: (jnp.int32(0),))
    return pl.pallas_call(
        functools.partial(_external_kernel, epsilon=epsilon),
        out_shape=jax.ShapeDtypeStruct(raw.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[matrix, matrix, vector, vector], out_specs=matrix, grid=(raw.shape[0] // bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret, name='external_mean_ln_remainder',
    )(raw, mean, scale, bias)


def _math(raw, scale, bias, epsilon):
    total = jnp.sum(raw, axis=1, keepdims=True)
    mean = (total / raw.shape[1]).astype(jnp.bfloat16).astype(jnp.float32)
    centered = raw.astype(jnp.bfloat16).astype(jnp.float32) - mean
    variance = jnp.sum(centered * centered, axis=1, keepdims=True) / raw.shape[1]
    invstd = jax.lax.rsqrt(variance + jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32))
    invstd = invstd.astype(jnp.bfloat16).astype(jnp.float32)
    result = jnp.maximum(centered * invstd * scale.astype(jnp.float32)[None, :]
                         + bias.astype(jnp.float32)[None, :], 0).astype(jnp.bfloat16)
    return tuple(jnp.broadcast_to(x, raw.shape).astype(jnp.float32)
                 for x in (total, mean, centered, variance, invstd, result))


def _kernel(raw, scale, bias, *outputs, epsilon):
    for ref, value in zip(outputs, _math(raw[...], scale[...], bias[...], epsilon)):
        ref[...] = value


def input_trace(raw, scale, bias, *, epsilon, pallas=False, interpret=False, bm=128):
    if raw.ndim != 2 or raw.dtype != jnp.float32 or raw.shape[0] % bm or raw.shape[1] % 128:
        raise ValueError('trace requires aligned float32 input')
    if not pallas:
        return _math(raw, scale, bias, epsilon)
    matrix = pl.BlockSpec((bm, raw.shape[1]), lambda i: (i.astype(jnp.int32), jnp.int32(0)))
    vector = pl.BlockSpec((raw.shape[1],), lambda i: (jnp.int32(0),))
    return pl.pallas_call(
        functools.partial(_kernel, epsilon=epsilon),
        out_shape=tuple(jax.ShapeDtypeStruct(raw.shape, jnp.float32) for _ in TRACE_NAMES),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, in_specs=[matrix, vector, vector],
            out_specs=tuple(matrix for _ in TRACE_NAMES), grid=(raw.shape[0] // bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret, name='input_raw_arithmetic_trace',
    )(raw, scale, bias)


def save_mismatch_rows(path, raw, reference, candidate):
    raw, reference, candidate = map(np.asarray, (raw, reference, candidate))
    coordinates = np.argwhere(reference != candidate)
    row_ids = np.unique(coordinates[:, 0])
    np.savez_compressed(path, coordinates=coordinates, row_ids=row_ids,
                        raw=raw[row_ids], reference=reference[row_ids], candidate=candidate[row_ids])
