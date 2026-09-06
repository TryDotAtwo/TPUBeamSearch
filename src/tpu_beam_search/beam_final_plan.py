"""Compose balanced destinations with source-routed final materialize requests."""
from .beam_final_balance import pallas_final_balance
from .beam_final_request import pallas_final_requests
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_packed_plan(packed, boundaries, *, world_size, interpret=False):
    """Adapt compacted [meta8,index2,valid1] records without reviving padding.

    Invalid records receive the exclusive terminal index K before balancing.
    Returned validity must still gate transport; this does not exchange requests.
    """
    if (packed.ndim != 2 or packed.shape[0] != 11
            or not packed.shape[1] or packed.shape[1] % 128
            or packed.dtype != jnp.uint32 or boundaries.shape != (2,128)
            or boundaries.dtype != jnp.uint32 or not 1 <= world_size <= 127):
        raise ValueError('invalid packed final plan ABI')

    def unpack(x, bounds, meta, indices):
        meta[...] = x[0:8,:]
        live = x[10,:] != 0
        indices[0,:] = jnp.where(live,x[8,:],bounds[0,world_size])
        indices[1,:] = jnp.where(live,x[9,:],bounds[1,world_size])

    n = packed.shape[1]
    meta, indices = pl.pallas_call(unpack,
        out_shape=(jax.ShapeDtypeStruct((8,n),jnp.uint32),
                   jax.ShapeDtypeStruct((2,n),jnp.uint32)),
        in_specs=(pl.BlockSpec((11,128),lambda i:(0,i)),
                  pl.BlockSpec((2,128),lambda i:(0,0))),
        out_specs=(pl.BlockSpec((8,128),lambda i:(0,i)),
                   pl.BlockSpec((2,128),lambda i:(0,i))),
        grid=(n//128,),interpret=interpret,
        name='beam_final_unpack_plan')(packed,boundaries)
    return pallas_final_plan(meta,indices,boundaries,
        world_size=world_size,interpret=interpret)


def pallas_final_plan(meta, indices, boundaries, *, world_size, interpret=False):
    """Return request words, source rank keys, and selected-row validity.

    Global indices already encode the source algorithm's less/equal phases
    and exact cap. Boundaries are agreed across ranks. This function does not
    choose winners or exchange data. Caller MUST compact by returned validity
    before sending; inactive request words are unspecified and must not be
    consumed. Parent/source/owner metadata is never rewritten by balancing.
    """
    if meta.ndim != 2 or indices.ndim != 2 or meta.shape[1] != indices.shape[1]:
        raise ValueError('final metadata and global indices must align')
    ranks, local, valid = pallas_final_balance(indices,boundaries,
        world_size=world_size,interpret=interpret)
    requests, sources = pallas_final_requests(meta,local,ranks,interpret=interpret)
    return requests,sources,valid
