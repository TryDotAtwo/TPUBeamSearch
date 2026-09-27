"""Frozen final selection through checked private frontier/history outputs."""
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from .beam_final_selection import make_final_selection_call
from .beam_final_delivery import pallas_final_delivery_plan
from .beam_final_delivery_exchange import pallas_prepare_delivery_exchange
from .beam_final_materialization_epochs import make_final_materialization_epochs


def make_final_depth_call(mesh, *, state_len, move_count, interpret=False):
    """Execute final selection, routing and both materialization streams.

    The resident inputs must already be frozen after S4/S5 work and DMA drains.
    The returned state remains private until the host publication boundary
    accepts it. This function does not construct or reuse scratch arenas.
    """
    select=make_final_selection_call(mesh,interpret=interpret)

    def local_target(counts,out):
        rank=jnp.int32(0) if mesh.size==1 else lax.axis_index('core')
        out[0]=jnp.sum(jnp.where(jnp.arange(128)==rank,counts[0,:],
                                    jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
    target_call=pl.pallas_call(local_target,
        out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
        interpret=interpret,name='beam_final_local_target_count')

    def call(a,b,controls,threshold,beam,prior_error,parents,generators,state):
        packed,keep,error,counts=select(a,b,controls,threshold,beam,prior_error)
        plan=pallas_final_delivery_plan(packed,keep,error,
            world_size=mesh.size,interpret=interpret)
        prepared=pallas_prepare_delivery_exchange(plan,
            world_size=mesh.size,interpret=interpret)
        epochs=make_final_materialization_epochs(mesh,state_len=state_len,
            move_count=move_count,request_capacity=prepared[0].shape[1],
            history_capacity=prepared[2].shape[1],interpret=interpret)
        target_counts=plan.target_counts[0,:mesh.size]
        local_count=target_call(plan.target_counts)
        state,response_error,history_error=epochs(state,parents,generators,
            prepared,target_counts,local_count)
        return state,target_counts,keep,counts,response_error,history_error

    return call
