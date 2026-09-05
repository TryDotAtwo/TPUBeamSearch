"""Aliased histogram publication with explicit local DMA completion.

Caller must exclude concurrent writers/readers of the inactive slot; this is
not a global S5 snapshot protocol or proof of physical TPU alias allocation.
"""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pallas_commit_histogram(a,b,control,new,*,interpret=False):
    """control[0,0]=active histogram; control[1,0]=physical processing flag.

    The clean survivor write must already have completed. This operation waits
    for every inactive-histogram DMA before publishing control and returning.
    Inputs A/B/control alias outputs; callers must use returned array versions.
    """
    if (a.ndim != 2 or a.shape[0] != 1 or a.shape[1] % 128 or not a.shape[1]
            or b.shape != a.shape or new.shape != a.shape or control.shape != (2,128)
            or any(x.dtype != jnp.uint32 for x in (a,b,control,new))):
        raise ValueError('invalid histogram publication ABI')
    width = a.shape[1]
    def kernel(ai,bi,ci,source,ao,bo,co,buffer,state,sem):
        def copy(src,dst):
            op = pltpu.make_async_copy(src,dst,sem)
            op.start()
            op.wait()
        copy(ci,state)
        active = state[0,0]&jnp.uint32(1)
        def body(i,_):
            section = pl.ds(i*128,128)
            copy(source.at[:,section],buffer)
            @pl.when(active == 0)
            def write_b():
                copy(buffer,bo.at[:,section])
            @pl.when(active != 0)
            def write_a():
                copy(buffer,ao.at[:,section])
            return None
        lax.fori_loop(0,width//128,body,None)
        lanes = jnp.arange(128,dtype=jnp.int32)[None] == 0
        rows = jnp.arange(2,dtype=jnp.int32)[:,None]
        updated = jnp.where(rows == 0,active^jnp.uint32(1),jnp.uint32(0))
        state[...] = jnp.where(lanes,updated,state[...])
        copy(state,co)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (a,b,control)),
        in_specs=(hbm,hbm,hbm,hbm),out_specs=(hbm,hbm,hbm),
        input_output_aliases={0:0,1:1,2:2},
        scratch_shapes=(pltpu.VMEM((1,128),jnp.uint32),pltpu.VMEM((2,128),jnp.uint32),
                        pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_histogram_dma_commit')(a,b,control,new)
