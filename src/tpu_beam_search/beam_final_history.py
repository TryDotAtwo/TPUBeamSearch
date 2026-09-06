"""Pallas final history projection; transport and host publication are separate."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def pallas_final_history_records(meta,target_local,valid,*,interpret=False):
    """Emit parent low/high, original route, target local, validity as uint32 SoA.

    Destination rank comes from the same final plan used by materialization.
    Caller routes these records to that destination and discards invalid rows
    before RankHistoryStore publication. This does not initiate a host copy.
    """
    if (meta.ndim != 2 or meta.shape[0] != 8 or not meta.shape[1]
            or meta.shape[1]%128 or target_local.shape != (1,meta.shape[1])
            or valid.shape != target_local.shape
            or any(x.dtype != jnp.uint32 for x in (meta,target_local,valid))):
        raise ValueError('invalid final history SoA ABI')
    def kernel(m,t,v,out):
        live=v[0,:] != 0
        for destination,source in enumerate((4,5,7)):
            out[destination,:]=jnp.where(live,m[source,:],0)
        out[3,:]=jnp.where(live,t[0,:],0)
        out[4,:]=live.astype(jnp.uint32)
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((5,meta.shape[1]),jnp.uint32),
        in_specs=tuple(pl.BlockSpec((r,128),lambda i:(0,i)) for r in (8,1,1)),
        out_specs=pl.BlockSpec((5,128),lambda i:(0,i)),
        grid=(meta.shape[1]//128,),interpret=interpret,
        name='beam_final_history_records')(meta,target_local,valid)
