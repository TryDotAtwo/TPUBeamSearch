"""Diagnostic packing prefixes; control outputs are NOT transport controls."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_probe(*,selection,world_size,interpret=False):
    def call(payload,intervals,error,index):
        def kernel(source,ranges,chunk,prior,out,control):
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
                control[0,0,:]=jnp.where(lanes==0,start,jnp.where(lanes==1,count,jnp.where(lanes==2,offset,jnp.uint32(0))))
        return pl.pallas_call(kernel,
            out_shape=(jax.ShapeDtypeStruct((world_size,32,128),jnp.uint32),jax.ShapeDtypeStruct((world_size,2,128),jnp.uint32)),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),pl.BlockSpec((3,128)),pl.BlockSpec((1,)),pl.BlockSpec((1,128))),
            out_specs=(pl.BlockSpec((1,32,128),lambda r:(r,0,0)),pl.BlockSpec((1,2,128),lambda r:(r,0,0))),
            grid=(world_size,),interpret=interpret,name='packing_selection_probe' if selection else 'packing_control_probe')(payload,intervals,index,error)
    return call
