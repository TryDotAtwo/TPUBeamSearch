"""Exact rank-major phase prefixes from already exchanged local counts."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_prefixes(counts,*,world_size,interpret=False):
    """counts uint32[2,128]: less/equal counts indexed by source rank.

    Return bases[4,128] (less low/high, adjusted equal low/high) and totals
    [4,128] (global less low/high, global equal low/high) in lane0 only.
    Input must be a globally agreed rank-ordered snapshot; no collective here.
    Padding is ignored. Local phase counts fit uint32; world<=128.
    """
    if counts.shape != (2,128) or counts.dtype != jnp.uint32 or not 1<=world_size<=128:
        raise ValueError('invalid final prefix ABI')
    def add(x,y):
        lo = x[0]+y[0]
        return lo,x[1]+y[1]+(lo<x[0]).astype(jnp.uint32)
    def kernel(c,b,t):
        lanes = jnp.arange(128,dtype=jnp.int32)
        valid = lanes<world_size
        less_total = (jnp.uint32(0),jnp.uint32(0))
        for phase in range(2):
            values = jnp.where(valid,c[phase,:],jnp.uint32(0))
            lo,hi = lax.associative_scan(add,(values,jnp.zeros_like(values)))
            total = (lo[-1],hi[-1])
            exlo = lo-values
            exhi = hi-(lo<values).astype(jnp.uint32)
            if phase == 0:
                less_total = total
            else:
                exlo,exhi = add((exlo,exhi),less_total)
            b[2*phase,:] = jnp.where(valid,exlo,jnp.uint32(0))
            b[2*phase+1,:] = jnp.where(valid,exhi,jnp.uint32(0))
            t[2*phase,:] = jnp.where(lanes==0,total[0],jnp.uint32(0))
            t[2*phase+1,:] = jnp.where(lanes==0,total[1],jnp.uint32(0))
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((4,128),jnp.uint32) for _ in range(2)),
        in_specs=(pl.BlockSpec((2,128)),),out_specs=(pl.BlockSpec((4,128)),pl.BlockSpec((4,128))),
        interpret=interpret,name='beam_final_rank_prefixes')(counts)
