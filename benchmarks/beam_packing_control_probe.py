"""Diagnostic packing prefixes; control outputs are NOT transport controls."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_probe(*,selection,world_size,interpret=False,guard=False,transfer=None):
    def call(payload,intervals,error,index):
        def kernel(source,ranges,chunk,prior,out,control,*scratch):
            peer=pl.program_id(0)
            lanes=jnp.arange(128,dtype=jnp.uint32)
            starts,counts=ranges[0,:],ranges[1,:]
            bad=jnp.any((lanes<world_size)&((counts>2048)|(starts>jnp.uint32(2048)-counts))) | (ranges[2,0]!=0) | (prior[0,0]!=0)
            out[...] = jnp.zeros((1,32,128),jnp.uint32)
            control[...] = jnp.zeros((1,2,128),jnp.uint32)
            control[0,1,:]=jnp.where(lanes==0,bad.astype(jnp.uint32),jnp.uint32(0))
            if selection:
                offset=jnp.minimum(chunk[0],jnp.uint32(16))*jnp.uint32(128)
                def select(values):
                    return jnp.sum(jnp.where(lanes==peer,values,jnp.uint32(0)).astype(jnp.int32),dtype=jnp.int32).astype(jnp.uint32)
                start,count=select(starts),select(counts)
                # Expose each value: a dead predicate must not erase the probe.
                if not guard:
                    control[0,0,:]=jnp.where(lanes==0,start,jnp.where(lanes==1,count,jnp.where(lanes==2,offset,jnp.uint32(0))))
                else:
                    @pl.when((~bad)&(offset<count))
                    def geometry():
                        length=jnp.minimum(jnp.uint32(128),count-offset)
                        begin=start+offset
                        aligned=(begin//jnp.uint32(128)*jnp.uint32(128)).astype(jnp.int32)
                        shift=(begin%jnp.uint32(128)).astype(jnp.int32)
                        control[0,0,:]=jnp.where(lanes==0,length,
                            jnp.where(lanes==1,begin,jnp.where(lanes==2,
                            aligned.astype(jnp.uint32),jnp.where(lanes==3,
                            shift.astype(jnp.uint32),jnp.uint32(0)))))
                        if transfer:
                            staging,sem=scratch
                            staging[...] = jnp.zeros((32,256),jnp.uint32)
                            first=pltpu.make_async_copy(source.at[:,pl.ds(aligned,128)],staging.at[:,pl.ds(0,128)],sem)
                            first.start()
                            first.wait()
                            if transfer in ('second','row_copy','unmasked_gather','gather','bounded_gather','rank2_gather','split_gather'):
                                @pl.when(shift+length.astype(jnp.int32)>128)
                                def second_tile():
                                    second=pltpu.make_async_copy(source.at[:,pl.ds(aligned+128,128)],staging.at[:,pl.ds(128,128)],sem)
                                    second.start()
                                    second.wait()
                            if transfer in ('positions','clipped_positions'):
                                positions=jnp.arange(128,dtype=jnp.int32)+shift
                                if transfer=='clipped_positions':
                                    positions=jnp.clip(positions,0,255)
                                for plane in range(32):
                                    out[0,plane,:]=positions.astype(jnp.uint32)
                            elif transfer == 'row_copy':
                                for plane in range(32):
                                    out[0,plane,:]=staging[plane,pl.ds(0,128)]
                            elif transfer == 'split_gather':
                                positions=jnp.arange(128,dtype=jnp.int32)+shift
                                indices=jnp.broadcast_to((positions%128)[None,:],(32,128))
                                low=jnp.take_along_axis(staging[:,pl.ds(0,128)],indices,axis=1,mode='promise_in_bounds')
                                high=jnp.take_along_axis(staging[:,pl.ds(128,128)],indices,axis=1,mode='promise_in_bounds')
                                out[0,:,:]=jnp.where(positions[None,:]<128,low,high)
                            elif transfer == 'rank2_gather':
                                positions=jnp.arange(128,dtype=jnp.int32)+shift
                                indices=jnp.broadcast_to(jnp.clip(positions,0,255)[None,:],(32,128))
                                out[0,:,:]=jnp.take_along_axis(staging[...],indices,axis=1,mode='promise_in_bounds')
                            elif transfer in ('unmasked_gather','gather','bounded_gather'):
                                from tpu_beam_search.beam_stream2 import _take_clipped
                                positions=jnp.arange(128,dtype=jnp.int32)+shift
                                for plane in range(32):
                                    # Geometry bounds positions to 0..254, independent of payload.
                                    values=(jnp.take_along_axis(staging[plane,:],positions,axis=0,mode='promise_in_bounds')
                                            if transfer=='bounded_gather' else _take_clipped(staging[plane,:],positions))
                                    out[0,plane,:]=jnp.where(lanes<length,values,jnp.uint32(0)) if transfer=='gather' else values
                            else:
                                column=0 if transfer=='first' else 128
                                out[0,:,:]=staging[:,pl.ds(column,128)]
        return pl.pallas_call(kernel,
            out_shape=(jax.ShapeDtypeStruct((world_size,32,128),jnp.uint32),jax.ShapeDtypeStruct((world_size,2,128),jnp.uint32)),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),pl.BlockSpec((3,128)),pl.BlockSpec((1,)),pl.BlockSpec((1,128))),
            out_specs=(pl.BlockSpec((1,32,128),lambda r:(r,0,0)),pl.BlockSpec((1,2,128),lambda r:(r,0,0))),
            grid=(world_size,),scratch_shapes=(pltpu.VMEM((32,256),jnp.uint32),pltpu.SemaphoreType.DMA) if transfer else (),
            interpret=interpret,name='packing_selection_probe' if selection else 'packing_control_probe')(payload,intervals,index,error)
    return call
