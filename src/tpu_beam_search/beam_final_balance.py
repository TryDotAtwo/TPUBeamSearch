"""Final rank assignment using exact two-word global indices and boundaries."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_balance(indices, boundaries, *, world_size, interpret=False):
    """Return rank, local index and validity (each uint32[1,N]).

    Caller supplies globally agreed boundaries ceil(rank*keep/world), including
    the terminal keep count, as low/high words [2,128]. World is <=127;
    each target interval must fit uint32. Invalid/padded global indices must
    be >= keep; they produce all zeros. No cap, prefix collective or history
    exchange is performed here. Boundary construction is a separate contract.
    """
    if (not 1 <= world_size <= 127 or indices.ndim != 2 or indices.shape[0] != 2
            or not indices.shape[1] or indices.shape[1] % 128
            or boundaries.shape != (2,128)
            or indices.dtype != jnp.uint32 or boundaries.dtype != jnp.uint32):
        raise ValueError('invalid final balance ABI')
    def kernel(x,b,rank,local,valid):
        lo,hi = x[0,:],x[1,:]
        def ge(r):
            return (hi > b[1,r]) | ((hi == b[1,r]) & (lo >= b[0,r]))
        target = jnp.zeros((128,),jnp.uint32)
        start = jnp.zeros((128,),jnp.uint32)
        for r in range(world_size):
            take = ge(r)
            target = jnp.where(take,jnp.uint32(r),target)
            start = jnp.where(take,b[0,r],start)
        live = ~ge(world_size)
        rank[...] = jnp.where(live,target,0)[None,:]
        local[...] = jnp.where(live,lo-start,0)[None,:]
        valid[...] = live.astype(jnp.uint32)[None,:]
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((1,indices.shape[1]),jnp.uint32) for _ in range(3)),
        in_specs=(pl.BlockSpec((2,128),lambda i:(0,i)),pl.BlockSpec((2,128),lambda i:(0,0))),
        out_specs=tuple(pl.BlockSpec((1,128),lambda i:(0,i)) for _ in range(3)),
        grid=(indices.shape[1]//128,),interpret=interpret,name='beam_final_balance')(indices,boundaries)
