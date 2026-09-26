"""Frozen physical A/B traversal adapter for source-faithful final selection."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_resident_view(a,b,controls,prior_error,*,interpret=False):
    """Return copied metadata [2*S,8,C], clean counts and local error.

    Physical order is A0,B0,A1,B1, not all A followed by all B. No cross-buffer
    hash dedup is performed: the audited CUDA final path selects physical
    clean prefixes independently. Caller must first drain work and refresh
    each required physical prefix under the final-local schedule. Zero dirty/
    busy controls are necessary, not proof of DMA completion or histogram
    freshness. This copy is not a zero-copy alias or a lifetime transition.
    An error rejects all prefixes and must enter common error agreement.
    """
    if (a.ndim != 3 or not 1 <= a.shape[0] <= 64 or a.shape[1] != 8
            or not a.shape[2] or a.shape[2]%128 or b.shape != a.shape
            or controls.shape != (a.shape[0],8,128)
            or prior_error.shape != (1,128)
            or any(x.dtype != jnp.uint32 for x in (a,b,controls,prior_error))):
        raise ValueError('invalid frozen resident ABI')
    shards,_,capacity = a.shape
    def pack(ar,br,out):
        out[0,:,:] = ar[0,:,:]
        out[1,:,:] = br[0,:,:]
    meta = pl.pallas_call(pack,
        out_shape=jax.ShapeDtypeStruct((2*shards,8,capacity),jnp.uint32),
        in_specs=(pl.BlockSpec((1,8,128),lambda s,t:(s,0,t)),)*2,
        out_specs=pl.BlockSpec((2,8,128),lambda s,t:(s,0,t)),
        grid=(shards,capacity//128),interpret=interpret,
        name='beam_final_physical_sibling_order')(a,b)
    def check(c,p,clean,error):
        bad = p[0,0] != 0
        for shard in range(shards):
            bad |= ((c[shard,0,0] > capacity) | (c[shard,1,0] > capacity)
                    | (c[shard,2,0] != 0) | (c[shard,3,0] != 0)
                    | (c[shard,4,0] != 0) | (c[shard,5,0] != 0)
                    | (c[shard,7,0] != 0))
        lanes = jnp.arange(128)[None]
        clean[...] = jnp.zeros((1,128),jnp.uint32)
        for shard in range(shards):
            for sibling in range(2):
                clean[...] = jnp.where((lanes == 2*shard+sibling) & ~bad,
                                       c[shard,sibling,0],clean[...])
        error[...] = jnp.where(lanes == 0,bad.astype(jnp.uint32),jnp.uint32(0))
    clean,error = pl.pallas_call(check,
        out_shape=(jax.ShapeDtypeStruct((1,128),jnp.uint32),)*2,
        interpret=interpret,name='beam_final_frozen_prefix_gate')(controls,prior_error)
    return meta,clean,error
