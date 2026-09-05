"""Source-ordered final score phase marking over clean shard prefixes."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_phase_masks(scores,clean,threshold,*,interpret=False):
    """uint32[shards,capacity] -> uint32[2,shards,capacity] masks.

    Clean counts are padded [1,ceil(shards/128)*128], each <=capacity.
    Caller freezes clean shards and supplies exact uint32 score keys, not
    floating scores or histogram bins. No selection, compaction or cap here.
    """
    if (scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]
            or scores.shape[1]%128
            or clean.shape != (1,((scores.shape[0]+127)//128)*128)
            or threshold.shape != (1,)
            or any(x.dtype != jnp.uint32 for x in (scores,clean,threshold))):
        raise ValueError('invalid final phase ABI')
    def kernel(s,c,t,out):
        shard = pl.program_id(0)
        slot = pl.program_id(1)*128+jnp.arange(128,dtype=jnp.int32)
        live = slot.astype(jnp.uint32)<c[0,shard%128]
        out[0,0,:] = (live & (s[0,:]<t[0])).astype(jnp.uint32)
        out[1,0,:] = (live & (s[0,:]==t[0])).astype(jnp.uint32)
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((2,*scores.shape),jnp.uint32),
        in_specs=(pl.BlockSpec((1,128),lambda s,t:(s,t)),
                  pl.BlockSpec((1,128),lambda s,t:(0,s//128)),
                  pl.BlockSpec((1,),lambda s,t:(0,))),
        out_specs=pl.BlockSpec((2,1,128),lambda s,t:(0,s,t)),
        grid=(scores.shape[0],scores.shape[1]//128),interpret=interpret,
        name='beam_final_phase_masks')(scores,clean,threshold)
