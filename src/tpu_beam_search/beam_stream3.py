"""Bounded Pallas Stream3 split after dedup and owner computation.

Diagnostic bitonic implementation, not an HBM-scale partition. Caller supplies
valid owners (< world_size) and moves (low byte of route), count <= capacity.
Both destination buffers have full capacity; neutral tails remain invalid.
Counts/offsets use padded [1, round_up(world_size+1,128)] uint32 storage.
No host data-dependent operation or per-owner cap is used.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .beam_dedup import _sort


def pallas_stream3_split(words, owners, count, *, local_rank, world_size, interpret=False):
    if (not isinstance(world_size, int) or not 1 <= world_size <= 256
            or not isinstance(local_rank, int) or not 0 <= local_rank < world_size):
        raise ValueError('invalid static topology')
    if words.ndim != 2 or words.shape[0] != 8 or words.dtype != jnp.uint32:
        raise ValueError('words must be uint32 [8,N]')
    n = words.shape[1]
    if n < 128 or n > 4096 or n & (n - 1):
        raise ValueError('diagnostic split capacity must be power of two in [128,4096]')
    if owners.shape != (1, n) or owners.dtype != jnp.uint32:
        raise ValueError('owners must be uint32 [1,N]')
    if count.shape != (1,) or count.dtype != jnp.uint32:
        raise ValueError('count must be uint32 [1]')
    control_width = ((world_size + 128) // 128) * 128

    def kernel(w, o, c, local_out, remote_out, local_count, counts_out, offsets_out):
        index = jnp.arange(n, dtype=jnp.uint32)
        owner = o[0, :]
        valid = index < c[0]
        is_local = valid & (owner == local_rank)
        is_remote = valid & (owner != local_rank)
        route = jnp.uint32(local_rank << 16) | (owner << 8) | (w[7, :] & 255)
        records = jnp.concatenate((w[:7, :], route[None, :]), axis=0)
        # _sort plane 9 is descending validity; plane 10 preserves source order.
        local_data = jnp.concatenate((records, o[...], is_local[None, :].astype(jnp.uint32), index[None, :]))
        remote_data = jnp.concatenate((records, o[...], is_remote[None, :].astype(jnp.uint32), index[None, :]))
        local_data = _sort(local_data, (9, 10))
        remote_data = _sort(remote_data, (9, 8, 10))
        neutral = jnp.where(jnp.arange(8)[:, None] == 6, jnp.uint32(0xffffffff), jnp.uint32(0))
        local_out[...] = jnp.where(local_data[9:10] != 0, local_data[:8], neutral)
        remote_out[...] = jnp.where(remote_data[9:10] != 0, remote_data[:8], neutral)
        positions = jnp.arange(control_width, dtype=jnp.uint32)[None, :]
        local_total = jnp.sum(is_local.astype(jnp.int32)).astype(jnp.uint32)
        local_count[...] = jnp.where(positions == 0, local_total, jnp.uint32(0))
        counts = jnp.zeros((1, control_width), jnp.uint32)
        offsets = jnp.zeros((1, control_width), jnp.uint32)
        running = jnp.uint32(0)
        for peer in range(world_size):
            amount = jnp.sum((is_remote & (owner == peer)).astype(jnp.int32)).astype(jnp.uint32)
            counts = jnp.where(positions == peer, amount, counts)
            running = running + amount
            offsets = jnp.where(positions == peer + 1, running, offsets)
        counts_out[...] = counts
        offsets_out[...] = offsets

    shapes = ((8, n), (8, n), (1, control_width), (1, control_width), (1, control_width))
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(s, jnp.uint32) for s in shapes),
        in_specs=tuple(pl.BlockSpec(x.shape) for x in (words, owners, count)),
        out_specs=tuple(pl.BlockSpec(s) for s in shapes), grid=(),
        interpret=interpret, name='beam_stream3_split')(words, owners, count)
