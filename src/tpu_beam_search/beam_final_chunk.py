"""Bounded final peer-chunk packing from grouped HBM records; no transport."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_stream2 import _take_clipped


def pallas_pack_final_chunk(payload,intervals,chunk,*,world_size,interpret=False):
    """Emit [rank,planes,128] and [rank,2,128] count/error controls.

    Chunk index is dynamic uint32. Two aligned input tiles cover an unaligned
    interval; each DMA completes before reading scratch. Any invalid interval
    blocks all ranks in this local call. Caller still needs collective error
    agreement and coordinated chunk epochs before inter-device transport.
    """
    if (not isinstance(world_size,int) or not 1 <= world_size <= 128
            or payload.ndim != 2 or not payload.shape[0] or not payload.shape[1]
            or payload.shape[1]%128 or payload.shape[1]>=1<<31
            or intervals.shape!=(3,128) or chunk.shape!=(1,)
            or any(x.dtype!=jnp.uint32 for x in (payload,intervals,chunk))):
        raise ValueError('invalid final chunk ABI')
    planes,n=payload.shape
    def kernel(source,ranges,index,out,control,staging,sem):
        peer=pl.program_id(0)
        lanes=jnp.arange(128,dtype=jnp.uint32)
        starts,counts=ranges[0,:],ranges[1,:]
        bad=jnp.any((lanes<world_size)&((counts>n)|(starts>jnp.uint32(n)-counts))) | (ranges[2,0]!=0)
        out[...] = jnp.zeros((1,planes,128),jnp.uint32)
        control[...] = jnp.zeros((1,2,128),jnp.uint32)
        control[0,1,:] = jnp.where(lanes==0,bad.astype(jnp.uint32),jnp.uint32(0))
        offset=jnp.minimum(index[0],jnp.uint32(n//128))*jnp.uint32(128)
        start,count=starts[peer],counts[peer]
        @pl.when((~bad)&(offset<count))
        def pack():
            length=jnp.minimum(jnp.uint32(128),count-offset)
            begin=start+offset
            aligned=(begin//jnp.uint32(128)*jnp.uint32(128)).astype(jnp.int32)
            shift=(begin%jnp.uint32(128)).astype(jnp.int32)
            staging[...] = jnp.zeros((planes,256),jnp.uint32)
            first=pltpu.make_async_copy(source.at[:,pl.ds(aligned,128)],staging.at[:,pl.ds(0,128)],sem)
            first.start()
            first.wait()
            @pl.when(shift+length.astype(jnp.int32)>128)
            def second_tile():
                second=pltpu.make_async_copy(source.at[:,pl.ds(aligned+128,128)],staging.at[:,pl.ds(128,128)],sem)
                second.start()
                second.wait()
            positions=jnp.arange(128,dtype=jnp.int32)+shift
            for plane in range(planes):
                values=_take_clipped(staging[plane,:],positions)
                out[0,plane,:]=jnp.where(lanes<length,values,jnp.uint32(0))
            control[0,0,:]=jnp.where(lanes==0,length,jnp.uint32(0))
    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((world_size,planes,128),jnp.uint32),
                   jax.ShapeDtypeStruct((world_size,2,128),jnp.uint32)),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),pl.BlockSpec((3,128)),pl.BlockSpec((1,))),
        out_specs=(pl.BlockSpec((1,planes,128),lambda r:(r,0,0)),pl.BlockSpec((1,2,128),lambda r:(r,0,0))),
        grid=(world_size,),scratch_shapes=(pltpu.VMEM((planes,256),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_final_peer_chunk')(payload,intervals,chunk)
