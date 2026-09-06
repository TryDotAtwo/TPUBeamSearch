"""Final response scatter to aliased HBM frontier; unique targets required."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_final_response import pallas_unpack_response
from .beam_final_error_summary import pallas_final_error_summary


def pallas_scatter_final_responses(frontier,wire,count,*,state_len,interpret=False):
    """Reject whole batch on target overflow; caller ensures unique targets.

    Count must fit wire capacity. Frontier is exclusive until call completion.
    This does not detect duplicate/missing targets or coordinate remote ranks.
    """
    if (frontier.ndim != 2 or frontier.dtype != jnp.uint8 or not 0 < frontier.shape[0] < 0x7fffffff
            or frontier.shape[1] != wire.shape[1] or count.shape != (1,) or count.dtype != jnp.uint32):
        raise ValueError('invalid final scatter ABI')
    clean,targets = pallas_unpack_response(wire,state_len=state_len,interpret=interpret)
    n,width = wire.shape
    def bounds(t,c,out):
        index = pl.program_id(0).astype(jnp.uint32)*128+jnp.arange(128,dtype=jnp.uint32)
        out[...] = (((index[None] < c[0]) & (t[...] >= frontier.shape[0]))
                    | ((c[0] > n) & (index[None] == 0))).astype(jnp.uint32)
    reason = pl.pallas_call(bounds,out_shape=jax.ShapeDtypeStruct((1,n),jnp.uint32),
        in_specs=(pl.BlockSpec((1,128),lambda i:(0,i)),pl.BlockSpec((1,))),
        out_specs=pl.BlockSpec((1,128),lambda i:(0,i)),grid=(n//128,),
        interpret=interpret,name='beam_final_scatter_bounds')(targets,count)
    errors = pallas_final_error_summary(reason,interpret=interpret)
    def scatter(old,data,t,c,e,out,staging,sem):
        index = pl.program_id(0)
        @pl.when((index.astype(jnp.uint32) < c[0]) & (e[0,0] == 0))
        def write():
            target = jnp.sum(jnp.where(jnp.arange(128) == index%128,t[0],jnp.uint32(0)).astype(jnp.int32))
            load = pltpu.make_async_copy(data.at[pl.ds(index,1),:],staging,sem)
            load.start()
            load.wait()
            store = pltpu.make_async_copy(staging,out.at[pl.ds(target,1),:],sem)
            store.start()
            store.wait()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    result = pl.pallas_call(scatter,out_shape=jax.ShapeDtypeStruct(frontier.shape,jnp.uint8),
        in_specs=(hbm,hbm,pl.BlockSpec((1,128),lambda i:(0,i//128)),pl.BlockSpec((1,)),pl.BlockSpec((2,128))),
        out_specs=hbm,input_output_aliases={0:0},grid=(n,),
        scratch_shapes=(pltpu.VMEM((1,width),jnp.uint8),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_final_response_scatter')(frontier,clean,targets,count,errors)
    return result,errors
