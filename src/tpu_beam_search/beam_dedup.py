"""Fixed-capacity diagnostic Pallas sort/dedup for Stream3 and Stream4.

This VMEM bitonic baseline is for bounded batches, not a claim of a scalable
HBM sort. Larger shards require the separately validated external merge path.
No per-shard beam cap is applied. Inputs are already CandidateMeta SoA; Stream3
owner routing must occur only after this operation.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _columns(data, indices):
    return jnp.take_along_axis(data, jnp.broadcast_to(indices[None, :], data.shape),
                               axis=1, mode='promise_in_bounds')


def _sort(data, key_planes):
    n = data.shape[1]
    indices = jnp.arange(n)
    size = 2
    while size <= n:
        stride = size // 2
        while stride:
            partner = _columns(data, indices ^ stride)
            less = jnp.zeros((n,), jnp.bool_)
            equal = jnp.ones((n,), jnp.bool_)
            for plane in key_planes:
                a, b = data[plane], partner[plane]
                if plane == 9:  # explicit validity, valid records first
                    a, b = 1 - a, 1 - b
                less = less | (equal & (a < b))
                equal = equal & (a == b)
            want_min = ((indices & size) == 0) == ((indices & stride) == 0)
            swap = jnp.where(want_min, ~less & ~equal, less)
            data = jnp.where(swap[None, :], partner, data)
            stride //= 2
        size *= 2
    return data


def pallas_threshold_dedup(words, payload, count, threshold, *, mode, interpret=False):
    if words.ndim != 2 or words.shape[0] != 8 or words.dtype != jnp.uint32:
        raise ValueError('words must be uint32 [8,N]')
    n = words.shape[1]
    if n < 128 or n & (n - 1) or n > 4096:
        raise ValueError('diagnostic sort capacity must be a power of two in [128,4096]')
    if payload.shape != (1, n) or payload.dtype != jnp.uint32:
        raise ValueError('payload must be uint32 [1,N]')
    if any(a.shape != (1,) or a.dtype != jnp.uint32 for a in (count, threshold)):
        raise ValueError('count and threshold must be uint32 [1]')
    if mode not in ('stream3', 'stream4'):
        raise ValueError('mode must be stream3 or stream4')

    def kernel(w, p, c, t, out, out_count):
        indices = jnp.arange(n, dtype=jnp.uint32)
        valid = ((indices < c[0]) & (w[6, :] <= t[0])).astype(jnp.uint32)
        data = jnp.concatenate((w[...], p[...], valid[None, :], indices[None, :]), axis=0)
        keys = (9, 3, 2, 1, 0, 6, 8) if mode == 'stream3' else (9, 3, 2, 1, 0, 6, 5, 4, 7)
        data = _sort(data, keys)
        previous = _columns(data[:4], jnp.maximum(indices, 1) - 1)
        unique = (data[9] != 0) & ((indices == 0) | jnp.any(data[:4] != previous, axis=0))
        data = jnp.concatenate((data[:9], unique[None, :].astype(jnp.uint32),
                                indices[None, :]), axis=0)
        # Stable compaction by validity then position; no conflicting scatter.
        data = _sort(data, (9, 10))
        keep = data[9] != 0
        neutral = jnp.zeros((8, n), jnp.uint32).at[6].set(jnp.uint32(0xffffffff))
        out[...] = jnp.where(keep[None, :], data[:8], neutral)
        out_count[0] = jnp.sum(keep.astype(jnp.uint32))

    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((8, n), jnp.uint32), jax.ShapeDtypeStruct((1,), jnp.uint32)),
        in_specs=tuple(pl.BlockSpec(x.shape) for x in (words, payload, count, threshold)),
        out_specs=(pl.BlockSpec((8, n)), pl.BlockSpec((1,))),
        grid=(), interpret=interpret, name='beam_threshold_sort_dedup_' + mode,
    )(words, payload, count, threshold)
