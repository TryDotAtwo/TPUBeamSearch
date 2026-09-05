"""Bounded-tile final validation summary, no parent-memory access."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_error_summary(reasons,*,interpret=False):
    """Output lane0: invalid count, first index (UINT32_MAX if none).

    Input reasons must already mask padding; capacity fits signed32 for min.
    Ordered tiles share the output state, as in periodic threshold scanning.
    This is diagnostic aggregation, not a cross-rank stop/publication protocol.
    """
    if (reasons.ndim != 2 or reasons.shape[0] != 1 or not reasons.shape[1]
            or reasons.shape[1]%128 or reasons.shape[1] > 0x7fffffff
            or reasons.dtype != jnp.uint32):
        raise ValueError('invalid final error reason ABI')
    def kernel(r,out):
        lanes = jnp.arange(128,dtype=jnp.int32)
        @pl.when(pl.program_id(0) == 0)
        def initialize():
            out[0,:] = jnp.zeros((128,),jnp.uint32)
            out[1,:] = jnp.where(lanes == 0,jnp.uint32(0xffffffff),jnp.uint32(0))
        bad = r[0] != 0
        count = out[0,0]+jnp.sum(bad.astype(jnp.int32)).astype(jnp.uint32)
        index = pl.program_id(0)*128+lanes
        first = jnp.min(jnp.where(bad,index,jnp.int32(0x7fffffff))).astype(jnp.uint32)
        first = jnp.where(first == 0x7fffffff,jnp.uint32(0xffffffff),first)
        first = jnp.minimum(out[1,0],first)
        out[0,:] = jnp.where(lanes == 0,count,jnp.uint32(0))
        out[1,:] = jnp.where(lanes == 0,first,jnp.uint32(0))
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
        in_specs=(pl.BlockSpec((1,128),lambda i:(0,i)),),
        out_specs=pl.BlockSpec((2,128),lambda i:(0,0)),
        grid=(reasons.shape[1]//128,),interpret=interpret,
        name='beam_final_error_summary')(reasons)
