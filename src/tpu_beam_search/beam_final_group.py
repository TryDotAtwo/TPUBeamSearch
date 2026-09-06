"""Stable rank grouping via tiled HBM bitonic sort; no exchange or speed claim."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_external_sort import pallas_external_bitonic_sort


def pallas_group_final_records(payload,ranks,valid,*,interpret=False):
    """Return payload planes followed by rank, original ordinal, validity.

    Capacity must be a power of two >=128. Caller supplies validated ranks and
    aligned payload; invalid rows are never sent regardless of payload values.
    Request routing uses source rank; history/response routing uses destination.
    This baseline takes O(N log^2 N) traffic and needs separate HBM pass buffers.
    """
    if (payload.ndim != 2 or not payload.shape[0] or payload.shape[1]<128
            or payload.shape[1]&(payload.shape[1]-1) or payload.shape[1]>=1<<31
            or ranks.shape!=(1,payload.shape[1]) or valid.shape!=ranks.shape
            or any(x.dtype!=jnp.uint32 for x in (payload,ranks,valid))):
        raise ValueError('invalid final grouping ABI')
    p,n=payload.shape
    def prepare(x,r,v,out):
        out[:p,:]=x[...]
        out[p,:]=r[0,:]
        out[p+1,:]=jnp.arange(128,dtype=jnp.uint32)+jnp.uint32(pl.program_id(0)*128)
        out[p+2,:]=(v[0,:]!=0).astype(jnp.uint32)
    records=pl.pallas_call(prepare,
        out_shape=jax.ShapeDtypeStruct((p+3,n),jnp.uint32),
        in_specs=tuple(pl.BlockSpec((s,128),lambda i:(0,i)) for s in (p,1,1)),
        out_specs=pl.BlockSpec((p+3,128),lambda i:(0,i)),
        grid=(n//128,),interpret=interpret,name='beam_final_group_prepare')(payload,ranks,valid)
    return pallas_external_bitonic_sort(records,key_planes=(p+2,p,p+1),
        validity_plane=p+2,interpret=interpret)
