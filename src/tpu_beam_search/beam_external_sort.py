"""HBM-staged compare/exchange primitives for scalable candidate sorting.

Each pass touches aligned candidate tiles and writes a separate HBM buffer.
Composing the bitonic network remains a JAX orchestration responsibility; no
kernel materializes the whole candidate array in VMEM.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_mark_sorted_unique(data, *, tile_candidates=128, interpret=False):
    """Mark first valid Hash128 in globally sorted 11-plane records.

    Input must be valid-first and Hash128/score/payload sorted. Planes 9/10
    become survivor validity/global sorted position for stable compaction.
    The predecessor tile is read from the original immutable input, so a
    duplicate spanning any number of tiles is suppressed without scan state.
    """
    if data.ndim != 2 or data.shape[0] != 11 or data.dtype != jnp.uint32:
        raise ValueError('data must be uint32 [11,N]')
    n = data.shape[1]
    if (not isinstance(tile_candidates, int) or tile_candidates < 128
            or tile_candidates & (tile_candidates - 1)
            or n < tile_candidates or n % tile_candidates):
        raise ValueError('tile must be a power of two >=128 dividing capacity')

    def kernel(current_ref, previous_ref, output_ref):
        block = pl.program_id(0)
        local = jnp.arange(tile_candidates, dtype=jnp.int32)
        value = current_ref[...]
        prior_tile = previous_ref[...]
        predecessor = local - (local != 0).astype(jnp.int32)
        indices = jnp.broadcast_to(predecessor[None, :], (4, tile_candidates))
        previous = jnp.take_along_axis(value[:4], indices, axis=1)
        last_indices = jnp.full((4, tile_candidates), tile_candidates - 1,
                               dtype=jnp.int32)
        boundary = jnp.take_along_axis(prior_tile[:4], last_indices, axis=1)
        previous = jnp.where((local == 0)[None, :], boundary, previous)
        position = block * tile_candidates + local
        unique = (value[9] != 0) & ((position == 0)
                   | jnp.any(value[:4] != previous, axis=0))
        output_ref[...] = jnp.concatenate((value[:9],
            unique[None, :].astype(jnp.uint32),
            position[None, :].astype(jnp.uint32)), axis=0)

    shape = (11, tile_candidates)
    own = pl.BlockSpec(shape, lambda block: (0, block))
    prior = pl.BlockSpec(shape, lambda block: (0, jnp.maximum(block - 1, 0)))
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct(data.shape, jnp.uint32),
        in_specs=(own, prior), out_specs=own,
        grid=(n // tile_candidates,), interpret=interpret,
        name='beam_external_mark_unique')(data, data)


def pallas_compare_exchange_pass(data, *, key_planes, size, stride,
                                 tile_candidates=128, validity_plane=None,
                                 interpret=False):
    """Apply one global bitonic compare/exchange pass to uint32 SoA data."""
    if data.ndim != 2 or data.dtype != jnp.uint32:
        raise ValueError('data must be uint32 [planes,N]')
    planes, capacity = data.shape
    if (not isinstance(tile_candidates, int) or tile_candidates < 128
            or tile_candidates & (tile_candidates - 1)
            or capacity % tile_candidates):
        raise ValueError('tile must be a power of two >=128 dividing capacity')
    if (not isinstance(size, int) or not isinstance(stride, int)
            or size < 2 or size & (size - 1) or size > capacity
            or stride < 1 or stride & (stride - 1)
            or stride >= size):
        raise ValueError('size and stride must describe a valid bitonic pass')
    key_planes = tuple(key_planes)
    if (not key_planes or any(not isinstance(p, int) or p < 0 or p >= planes
                              for p in key_planes)):
        raise ValueError('key planes must index data')
    if validity_plane is not None and validity_plane not in key_planes:
        raise ValueError('validity plane must be one of the key planes')
    tile_shape = (planes, tile_candidates)
    partner_block_xor = stride // tile_candidates

    def kernel(self_ref, partner_ref, output_ref):
        block = pl.program_id(0)
        local_index = jnp.arange(tile_candidates, dtype=jnp.uint32)
        global_index = (jnp.uint32(block * tile_candidates) + local_index)
        value = self_ref[...]
        if stride < tile_candidates:
            partner_index = local_index ^ jnp.uint32(stride)
            gather_index = jnp.broadcast_to(partner_index[None, :], value.shape)
            partner = jnp.take_along_axis(value, gather_index, axis=1)
        else:
            partner = partner_ref[...]
        less = jnp.zeros((tile_candidates,), jnp.bool_)
        equal = jnp.ones((tile_candidates,), jnp.bool_)
        for plane in key_planes:
            a, b = value[plane], partner[plane]
            if plane == validity_plane:
                a, b = jnp.uint32(1) - a, jnp.uint32(1) - b
            less = less | (equal & (a < b))
            equal = equal & (a == b)
        want_min = (((global_index & jnp.uint32(size)) == 0)
                    == ((global_index & jnp.uint32(stride)) == 0))
        swap = (want_min & ~less & ~equal) | (~want_min & less)
        output_ref[...] = jnp.where(swap[None, :], partner, value)

    own_spec = pl.BlockSpec(tile_shape, lambda block: (0, block))
    partner_spec = pl.BlockSpec(
        tile_shape,
        lambda block: (0, block ^ partner_block_xor)
        if stride >= tile_candidates else (0, block))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(data.shape, jnp.uint32),
        in_specs=(own_spec, partner_spec),
        out_specs=own_spec,
        grid=(capacity // tile_candidates,),
        interpret=interpret,
        name=f'beam_external_bitonic_s{size}_d{stride}',
    )(data, data)


def pallas_external_bitonic_sort(data, *, key_planes, tile_candidates=128,
                                 validity_plane=None, interpret=False):
    """Compose aligned HBM compare/exchange passes into a global sort."""
    if data.ndim != 2:
        raise ValueError('data must be rank two')
    capacity = data.shape[1]
    if capacity < 2 or capacity & (capacity - 1):
        raise ValueError('capacity must be a power of two')
    result = data
    size = 2
    while size <= capacity:
        stride = size // 2
        while stride:
            result = pallas_compare_exchange_pass(
                result, key_planes=key_planes, size=size, stride=stride,
                tile_candidates=tile_candidates,
                validity_plane=validity_plane, interpret=interpret)
            stride //= 2
        size *= 2
    return result


def pallas_external_stream3_dedup(words, payload, count, threshold, *,
                                  tile_candidates=128, interpret=False):
    """HBM-staged S3 threshold/sort/unique/compact baseline, before routing.

    Count must be in [0,N]; payload IDs identify source records. No beam cap
    is applied. Bitonic compaction is deliberately a correctness baseline.
    The temporary limit bounds the separate count-reduction control window.
    """
    if words.ndim != 2 or words.shape[0] != 8 or words.dtype != jnp.uint32:
        raise ValueError('words must be uint32 [8,N]')
    n = words.shape[1]
    if (n < 128 or n > 16384 or n & (n - 1)
            or tile_candidates != 128):
        raise ValueError('capacity must be power of two in [128,16384], tile=128')
    if payload.shape != (1, n) or payload.dtype != jnp.uint32:
        raise ValueError('payload must be uint32 [1,N]')
    if any(x.shape != (1,) or x.dtype != jnp.uint32 for x in (count, threshold)):
        raise ValueError('count and threshold must be uint32 [1]')
    tiles = n // tile_candidates
    word_spec = pl.BlockSpec((8, tile_candidates), lambda b: (0, b))
    data_spec = pl.BlockSpec((11, tile_candidates), lambda b: (0, b))

    def prepare(w, p, c, t, out):
        index = (pl.program_id(0) * tile_candidates
                 + jnp.arange(tile_candidates, dtype=jnp.int32)).astype(jnp.uint32)
        value = w[...]
        valid = (index < c[0]) & (value[6] <= t[0])
        out[...] = jnp.concatenate((value, p[...],
            valid[None, :].astype(jnp.uint32), index[None, :]), axis=0)

    data = pl.pallas_call(prepare,
        out_shape=jax.ShapeDtypeStruct((11, n), jnp.uint32),
        in_specs=(word_spec, pl.BlockSpec((1, tile_candidates), lambda b: (0, b)),
                  pl.BlockSpec((1,), lambda b: (0,)),
                  pl.BlockSpec((1,), lambda b: (0,))),
        out_specs=data_spec, grid=(tiles,), interpret=interpret,
        name='beam_external_threshold')(words, payload, count, threshold)
    data = pallas_external_bitonic_sort(data,
        key_planes=(9, 3, 2, 1, 0, 6, 8, 10), validity_plane=9,
        tile_candidates=tile_candidates, interpret=interpret)
    data = pallas_mark_sorted_unique(data, tile_candidates=tile_candidates,
                                     interpret=interpret)
    data = pallas_external_bitonic_sort(data, key_planes=(9, 10),
        validity_plane=9, tile_candidates=tile_candidates, interpret=interpret)

    def finish(records, out, counts):
        value = records[...]
        keep = value[9] != 0
        neutral = jnp.where(jnp.arange(8)[:, None] == 6,
                            jnp.uint32(0xffffffff), jnp.uint32(0))
        out[...] = jnp.where(keep[None, :], value[:8], neutral)
        total = jnp.sum(keep.astype(jnp.int32)).astype(jnp.uint32)
        counts[...] = (jnp.arange(128)[None, :] == 0).astype(jnp.uint32) * total

    result, counts = pl.pallas_call(finish,
        out_shape=(jax.ShapeDtypeStruct((8, n), jnp.uint32),
                   jax.ShapeDtypeStruct((tiles, 128), jnp.uint32)),
        in_specs=(data_spec,),
        out_specs=(word_spec, pl.BlockSpec((1, 128), lambda b: (b, 0))),
        grid=(tiles,), interpret=interpret, name='beam_external_neutral_counts')(data)

    def total_count(c, out):
        total = jnp.sum(c[...].astype(jnp.int32)).astype(jnp.uint32)
        out[...] = (jnp.arange(128)[None, :] == 0).astype(jnp.uint32) * total

    total = pl.pallas_call(total_count,
        out_shape=jax.ShapeDtypeStruct((1, 128), jnp.uint32),
        in_specs=(pl.BlockSpec(counts.shape),), out_specs=pl.BlockSpec((1, 128)),
        grid=(), interpret=interpret, name='beam_external_total_count')(counts)
    return result, total
