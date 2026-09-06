"""Cap local phase ordinals with exact global pair-word prefixes."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_indices(ordinal,bases,keep,error,*,rank,interpret=False):
    """Return global indices [phase,word,shard,slot] and uint32 validity.

    Inputs come from phase_scan/prefixes/cap for the same frozen epoch.
    UINT32_MAX ordinals are invalid. Any cap error rejects every lane.
    Prefix+ordinal must fit uint64. Zero invalid index words are unspecified
    destinations: downstream MUST compact/gate by validity before balance.
    """
    if (ordinal.ndim != 3 or ordinal.shape[0] != 2 or not ordinal.shape[1]
            or not ordinal.shape[2] or ordinal.shape[2]%128
            or bases.shape != (4,128) or keep.shape != (2,128)
            or error.shape != (1,128) or not 0<=rank<128
            or any(x.dtype != jnp.uint32 for x in (ordinal,bases,keep,error))):
        raise ValueError('invalid final index ABI')
    def kernel(o,b,k,e,index,valid):
        phase = pl.program_id(0)
        base = b[2*phase,rank]
        lo = base+o[0,0,:]
        hi = b[2*phase+1,rank]+(lo<base).astype(jnp.uint32)
        below = (hi<k[1,0])|((hi==k[1,0])&(lo<k[0,0]))
        live = (o[0,0,:]!=jnp.uint32(0xffffffff))&below&(e[0,0]==0)
        index[0,0,0,:] = jnp.where(live,lo,jnp.uint32(0))
        index[0,1,0,:] = jnp.where(live,hi,jnp.uint32(0))
        valid[0,0,:] = live.astype(jnp.uint32)
    spec = pl.BlockSpec((1,1,128),lambda p,s,t:(p,s,t))
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((2,2,*ordinal.shape[1:]),jnp.uint32),
                   jax.ShapeDtypeStruct(ordinal.shape,jnp.uint32)),
        in_specs=(spec,pl.BlockSpec((4,128),lambda p,s,t:(0,0)),
                  pl.BlockSpec((2,128),lambda p,s,t:(0,0)),pl.BlockSpec((1,128),lambda p,s,t:(0,0))),
        out_specs=(pl.BlockSpec((1,2,1,128),lambda p,s,t:(p,0,s,t)),spec),
        grid=(2,ordinal.shape[1],ordinal.shape[2]//128),interpret=interpret,
        name='beam_final_capped_indices')(ordinal,bases,keep,error)
