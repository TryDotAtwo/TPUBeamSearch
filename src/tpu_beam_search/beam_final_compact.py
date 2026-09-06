"""HBM bitonic final compaction baseline; not a linear-time compactor."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_external_sort import pallas_external_bitonic_sort


def pallas_final_compact(meta,indices,valid,*,interpret=False):
    """Return [11,power-of-two capacity]: meta8, global-index2, validity1.

    Inputs preserve [shard,slot] traversal. Selected global indices must be
    unique and originate from capped phases of one frozen epoch. No record
    is dropped for capacity: allocate next-power-of-two(2*shards*capacity).
    Invalid output records are zero. All passes are tile-staged in HBM;
    O(N log^2 N) work and doubled candidate scratch are explicit baseline costs.
    """
    if (meta.ndim!=3 or meta.shape[0]!=8 or not meta.shape[1] or not meta.shape[2]
            or meta.shape[2]%128 or indices.shape!=(2,2,*meta.shape[1:])
            or valid.shape!=(2,*meta.shape[1:])
            or any(x.dtype!=jnp.uint32 for x in (meta,indices,valid))):
        raise ValueError('invalid final compaction ABI')
    shards,capacity=meta.shape[1:]
    n=1<<(2*shards*capacity-1).bit_length()
    if n>=0x80000000:
        raise ValueError('final compaction tile indices exceed signed32')
    tiles=capacity//128
    def prepare(m,i,v,out):
        tile=pl.program_id(0)
        live=(tile<2*shards*tiles)&(v[0,0,:]!=0)
        out[...] = jnp.concatenate((m[:,0,:],i[0,:,0,:],live.astype(jnp.uint32)[None,:]))
    prepared=pl.pallas_call(prepare,out_shape=jax.ShapeDtypeStruct((11,n),jnp.uint32),
        in_specs=(pl.BlockSpec((8,1,128),lambda t:(0,(t//tiles)%shards,t%tiles)),
                  pl.BlockSpec((1,2,1,128),lambda t:(jnp.minimum(t//(shards*tiles),1),0,(t//tiles)%shards,t%tiles)),
                  pl.BlockSpec((1,1,128),lambda t:(jnp.minimum(t//(shards*tiles),1),(t//tiles)%shards,t%tiles))),
        out_specs=pl.BlockSpec((11,128),lambda t:(0,t)),grid=(n//128,),
        interpret=interpret,name='beam_final_compact_prepare')(meta,indices,valid)
    sorted_data=pallas_external_bitonic_sort(prepared,key_planes=(10,9,8),validity_plane=10,interpret=interpret)
    def clear(x,out):
        out[...] = jnp.where((x[10,:]!=0)[None,:],x[...],jnp.uint32(0))
    spec=pl.BlockSpec((11,128),lambda t:(0,t))
    return pl.pallas_call(clear,out_shape=jax.ShapeDtypeStruct((11,n),jnp.uint32),
        in_specs=(spec,),out_specs=spec,grid=(n//128,),interpret=interpret,
        name='beam_final_compact_clear')(sorted_data)
