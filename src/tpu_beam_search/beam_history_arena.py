"""History projection directly between nonoverlapping regions of one arena."""
from numbers import Integral
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pallas_history_in_arena(arena,target,valid,*,meta_offset,history_offset,interpret=False):
    """Read flattened uint32[8,N] metadata, write flattened uint32[5,N] history.

    Offsets are bytes aligned512. Preserve all other arena words. Output aliases
    arena inside Pallas; outer JIT donation and consumer lifetime are caller
    obligations. No host history publication or transfer occurs here.
    """
    if (arena.ndim!=2 or arena.shape[1]!=128 or arena.dtype!=jnp.uint32
            or target.ndim!=2 or target.shape[0]!=1 or not target.shape[1]
            or target.shape[1]%128 or valid.shape!=target.shape
            or target.dtype!=jnp.uint32 or valid.dtype!=jnp.uint32
            or any(isinstance(x,bool) or not isinstance(x,Integral) or x<0 or x%512
                   for x in (meta_offset,history_offset))):
        raise ValueError('invalid history arena ABI')
    n=target.shape[1]
    mend,hend=meta_offset+8*n*4,history_offset+5*n*4
    if max(mend,hend)>arena.size*4 or (meta_offset<hend and history_offset<mend):
        raise ValueError('history arena regions overlap or exceed capacity')
    tiles=n//128
    mstart,hstart=int(meta_offset)//512,int(history_offset)//512
    def kernel(old,t,v,out,stage,sem):
        tile=pl.program_id(0)
        live=v[...]!=0
        for destination,source in enumerate((4,5,7)):
            load=pltpu.make_async_copy(old.at[pl.ds(mstart+source*tiles+tile,1),:],stage,sem)
            load.start()
            load.wait()
            stage[...] = jnp.where(live,stage[...],jnp.uint32(0))
            store=pltpu.make_async_copy(stage,out.at[pl.ds(hstart+destination*tiles+tile,1),:],sem)
            store.start()
            store.wait()
        stage[...] = jnp.where(live,t[...],jnp.uint32(0))
        store=pltpu.make_async_copy(stage,out.at[pl.ds(hstart+3*tiles+tile,1),:],sem)
        store.start()
        store.wait()
        stage[...] = live.astype(jnp.uint32)
        store=pltpu.make_async_copy(stage,out.at[pl.ds(hstart+4*tiles+tile,1),:],sem)
        store.start()
        store.wait()
    hbm=pl.BlockSpec(memory_space=pltpu.HBM)
    vec=pl.BlockSpec((1,128),lambda i:(0,i))
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(arena.shape,jnp.uint32),
        in_specs=(hbm,vec,vec),out_specs=hbm,input_output_aliases={0:0},grid=(tiles,),
        scratch_shapes=(pltpu.VMEM((1,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_history_arena')(arena,target,valid)
