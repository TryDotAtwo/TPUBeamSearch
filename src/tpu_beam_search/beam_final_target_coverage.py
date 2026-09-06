"""Exact whole-final target coverage gate, using existing tiled HBM sorting."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_external_sort import pallas_external_bitonic_sort


def pallas_final_target_coverage(targets, valid, target_count, *, interpret=False):
    """Return nonzero reasons unless live targets are exactly range(target_count).

    Caller supplies all local final targets across chunks, not one partial
    chunk. Sorting is diagnostic O(N log^2 N) HBM traffic. This is not a
    distributed publication barrier; aggregate errors before publishing.
    """
    if (targets.ndim != 2 or targets.shape[0] != 1 or targets.shape[1] < 128
            or targets.shape[1] & (targets.shape[1]-1) or targets.shape[1] >= 1 << 31
            or valid.shape != targets.shape or target_count.shape != (1,)
            or any(x.dtype != jnp.uint32 for x in (targets,valid,target_count))):
        raise ValueError('invalid final coverage ABI')
    n = targets.shape[1]
    tile = pl.BlockSpec((1,128),lambda i:(0,i))
    def prepare(t,v,out):
        out[0,:] = t[0,:]
        out[1,:] = (v[0,:] != 0).astype(jnp.uint32)
    records = pl.pallas_call(prepare,out_shape=jax.ShapeDtypeStruct((2,n),jnp.uint32),
        in_specs=(tile,tile),out_specs=pl.BlockSpec((2,128),lambda i:(0,i)),
        grid=(n//128,),interpret=interpret,name='beam_final_coverage_prepare')(targets,valid)
    ordered = pallas_external_bitonic_sort(records,key_planes=(1,0),validity_plane=1,interpret=interpret)
    def check(r,c,out):
        index = jnp.uint32(pl.program_id(0)*128)+jnp.arange(128,dtype=jnp.uint32)
        wanted = index < c[0]
        bad = (wanted != (r[1,:] != 0)) | (wanted & (r[0,:] != index))
        bad |= (c[0] > n) & (index == 0)
        out[...] = bad.astype(jnp.uint32)[None,:]
    return pl.pallas_call(check,out_shape=jax.ShapeDtypeStruct((1,n),jnp.uint32),
        in_specs=(pl.BlockSpec((2,128),lambda i:(0,i)),pl.BlockSpec((1,))),
        out_specs=tile,grid=(n//128,),interpret=interpret,
        name='beam_final_target_coverage')(ordered,target_count)
