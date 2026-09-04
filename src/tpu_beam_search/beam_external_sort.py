"""HBM-staged compare/exchange primitives for scalable candidate sorting.

Each pass touches aligned candidate tiles and writes a separate HBM buffer.
Composing the bitonic network remains a JAX orchestration responsibility; no
kernel materializes the whole candidate array in VMEM.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


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
