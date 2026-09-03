"""Controlled consumers of a materialized variance buffer, not a model engine."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def collect_consumer(values, operation, *, devices=8, chunk_rows=256, width=1024):
    from benchmarks.artgor_prefix_shape import chunked_host
    result = chunked_host(values, operation, devices=devices, chunk_rows=chunk_rows)
    if result.ndim == 1:
        return np.broadcast_to(result[:, None], (len(result), width))
    if result.shape != (len(values), width):
        raise ValueError('unexpected consumer output shape')
    return result


def _expression(values, *, arithmetic, epsilon):
    if arithmetic == 'bf16_expression':
        # Source-level BF16 contract; TPU lowering may elide intermediates.
        return jax.lax.rsqrt(values.astype(jnp.bfloat16) + epsilon)
    return jax.lax.rsqrt(values.astype(jnp.float32)
        + jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)


def _kernel(values, output, *, arithmetic, epsilon):
    output[...] = _expression(values[...], arithmetic=arithmetic, epsilon=epsilon)


def consume_variance(values, *, engine, arithmetic='fp32', epsilon=1e-5,
                     interpret=False):
    """Preserve scalar-1D or broadcast-2D storage and return BF16 invstd."""
    if engine not in ('jax', 'pallas') or arithmetic not in ('fp32', 'bf16_expression'):
        raise ValueError('unknown consumer engine/arithmetic')
    if engine == 'jax':
        return _expression(values, arithmetic=arithmetic, epsilon=epsilon)
    if values.ndim == 1:
        block = pl.BlockSpec((128,), lambda i: (i.astype(jnp.int32),))
    elif values.ndim == 2:
        block = pl.BlockSpec((128, values.shape[1]),
                            lambda i: (i.astype(jnp.int32), jnp.int32(0)))
    else:
        raise ValueError('consumer expects rank one or two')
    if values.shape[0] % 128:
        raise ValueError('consumer rows must be aligned to 128')
    return pl.pallas_call(functools.partial(_kernel, arithmetic=arithmetic, epsilon=epsilon),
        out_shape=jax.ShapeDtypeStruct(values.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[block], out_specs=block, grid=(values.shape[0] // 128,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret, name=f'rsqrt_{arithmetic}_{values.ndim}d')(values)
