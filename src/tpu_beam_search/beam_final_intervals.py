"""Rank intervals for an already stable-grouped, valid-prefix final buffer."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_rank_intervals(ranks,valid,*,world_size,interpret=False):
    """Return uint32[3,128]: exclusive starts, counts, bad-rank count in lane0.

    Caller supplies the rank/valid planes of pallas_group_final_records.
    Live records must form a prefix sorted by rank. Any nonzero error blocks
    transport collectively. Invalid padding ranks are ignored. This ordered
    tiled reduction is not a parallel grid or a physical TPU acceptance result.
    """
    if (not isinstance(world_size,int) or not 1 <= world_size <= 128
            or ranks.ndim != 2 or ranks.shape[0] != 1 or ranks.shape[1] == 0
            or ranks.shape[1]%128 or ranks.shape[1] >= 1<<31
            or valid.shape != ranks.shape
            or ranks.dtype != jnp.uint32 or valid.dtype != jnp.uint32):
        raise ValueError('invalid final rank interval ABI')
    tiles = ranks.shape[1]//128
    def kernel(r,v,out):
        tile = pl.program_id(0)
        lanes = jnp.arange(128,dtype=jnp.uint32)
        @pl.when(tile == 0)
        def initialize():
            out[...] = jnp.zeros((3,128),jnp.uint32)
        live = v[0,:] != 0
        hits = (lanes[:,None] == r[0,:][None,:]) & live[None,:]
        counts = jnp.sum(hits.astype(jnp.uint32),axis=1)
        out[1,:] = out[1,:]+jnp.where(lanes < world_size,counts,jnp.uint32(0))
        bad = jnp.sum((live & (r[0,:] >= world_size)).astype(jnp.uint32))
        out[2,:] = out[2,:]+jnp.where(lanes == 0,bad,jnp.uint32(0))
        @pl.when(tile == tiles-1)
        def finalize():
            counts = out[1,:]
            starts = lax.associative_scan(jnp.add,counts)-counts
            out[0,:] = jnp.where(lanes < world_size,starts,jnp.uint32(0))
    spec = pl.BlockSpec((1,128),lambda i:(0,i))
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((3,128),jnp.uint32),
        in_specs=(spec,spec),out_specs=pl.BlockSpec((3,128),lambda i:(0,0)),
        grid=(tiles,),interpret=interpret,name='beam_final_rank_intervals')(ranks,valid)
