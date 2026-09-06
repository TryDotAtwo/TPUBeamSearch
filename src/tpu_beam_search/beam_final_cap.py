"""Exact global final count and source threshold invariant, no collectives."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_cap(totals,beam,*,interpret=False):
    """Return K=min(beam,L+E) pair and error lane0 if L>K.

    totals[4,128] has L low/high then E low/high in lane0; beam[2,128].
    Sum must fit uint64. Any error MUST block selection/materialization;
    K is diagnostic on error, not permission to truncate the less phase.
    """
    if (totals.shape != (4,128) or beam.shape != (2,128)
            or totals.dtype != jnp.uint32 or beam.dtype != jnp.uint32):
        raise ValueError('invalid final cap ABI')
    def kernel(t,b,k,e):
        lo = t[0,0]+t[2,0]
        hi = t[1,0]+t[3,0]+(lo<t[0,0]).astype(jnp.uint32)
        use_total = (hi<b[1,0])|((hi==b[1,0])&(lo<=b[0,0]))
        kl,kh = jnp.where(use_total,lo,b[0,0]),jnp.where(use_total,hi,b[1,0])
        bad = (t[1,0]>kh)|((t[1,0]==kh)&(t[0,0]>kl))
        lane0 = jnp.arange(128)==0
        k[0,:],k[1,:] = jnp.where(lane0,kl,jnp.uint32(0)),jnp.where(lane0,kh,jnp.uint32(0))
        e[0,:] = jnp.where(lane0,bad.astype(jnp.uint32),jnp.uint32(0))
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((2,128),jnp.uint32),jax.ShapeDtypeStruct((1,128),jnp.uint32)),
        in_specs=(pl.BlockSpec((4,128)),pl.BlockSpec((2,128))),
        out_specs=(pl.BlockSpec((2,128)),pl.BlockSpec((1,128))),
        interpret=interpret,name='beam_final_exact_cap')(totals,beam)
