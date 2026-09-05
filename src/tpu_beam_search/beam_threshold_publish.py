"""Serialized double-slot periodic threshold publication, not an S5 epoch."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pallas_publish_periodic_threshold(a,b,active,candidate,*,interpret=False):
    """Slots/candidate lane0 contain threshold and initialized; active[0,0].

    The caller owns publication and excludes readers of the inactive slot until
    their captured reads finish. Return aliased new versions. This local DMA
    sequence does not coordinate ranks or protect readers across two publishes.
    Final selection uses a different policy and must not call this routine.
    """
    if (a.shape != (2,128) or b.shape != a.shape or candidate.shape != a.shape
            or active.shape != (1,128)
            or any(x.dtype != jnp.uint32 for x in (a,b,active,candidate))):
        raise ValueError('invalid periodic threshold publication ABI')
    def kernel(ai,bi,ci,ni,ao,bo,co,state,old,publish,new,sem):
        def copy(src,dst):
            op = pltpu.make_async_copy(src,dst,sem)
            op.start()
            op.wait()
        copy(ci,state)
        selected = state[0,0]&jnp.uint32(1)
        @pl.when(selected == 0)
        def load_a():
            copy(ai,old)
            copy(bi,publish)
        @pl.when(selected != 0)
        def load_b():
            copy(bi,old)
            copy(ai,publish)
        copy(ni,new)
        initialized = old[1,0] != 0
        valid = new[1,0] != 0
        value = jnp.where(valid,jnp.where(initialized,jnp.minimum(old[0,0],new[0,0]),new[0,0]),
                          jnp.where(initialized,old[0,0],jnp.uint32(0xffffffff)))
        rows = jnp.arange(2,dtype=jnp.int32)[:,None]
        lane0 = jnp.arange(128,dtype=jnp.int32)[None] == 0
        publish[...] = jnp.where(lane0,jnp.where(rows == 0,value,
            (initialized|valid).astype(jnp.uint32)),publish[...])
        @pl.when(selected == 0)
        def write_b():
            copy(publish,bo)
        @pl.when(selected != 0)
        def write_a():
            copy(publish,ao)
        # Inactive value/init DMA has completed before the control switch.
        state[...] = jnp.where(lane0,selected^jnp.uint32(1),state[...])
        copy(state,co)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (a,b,active)),
        in_specs=(hbm,)*4,out_specs=(hbm,)*3,input_output_aliases={0:0,1:1,2:2},
        scratch_shapes=(pltpu.VMEM((1,128),jnp.uint32),
            pltpu.VMEM((2,128),jnp.uint32),pltpu.VMEM((2,128),jnp.uint32),
            pltpu.VMEM((2,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_periodic_threshold_publish')(a,b,active,candidate)
