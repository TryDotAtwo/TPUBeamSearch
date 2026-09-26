"""One serialized paired-delivery epoch and its bounded response subepochs."""
from typing import NamedTuple
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_delivery_exchange import make_delivery_chunk_call
from .beam_final_history_consumer import make_final_history_consumer
from .beam_final_response_plan import pallas_final_response_plan
from .beam_final_response_chunk import make_final_response_chunk_call
from .beam_final_response_consumer import make_final_response_consumer


class FinalMaterializationState(NamedTuple):
    frontier: jax.Array
    history: jax.Array
    response_marks: jax.Array
    response_control: jax.Array
    history_marks: jax.Array
    history_control: jax.Array
    error: jax.Array


def make_final_materialization_round(mesh,*,state_len,move_count,interpret=False):
    """All ranks call the same request epochs and world response subepochs.

    A request epoch receives at most world*128 records at each source, so world
    response subepochs cover even a single concentrated return destination.
    The conservative uniform schedule differs from CUDA variable-size exchange;
    it makes no overlap or efficiency claim. Private arenas persist across
    calls. Old parent frontier/generators remain live through materialization.
    Final two-stream coverage, publication and scratch reuse are caller-owned.
    """
    delivery=make_delivery_chunk_call(mesh,interpret=interpret)
    consume_history=make_final_history_consumer(mesh,move_count=move_count,interpret=interpret)
    consume_response=make_final_response_consumer(mesh,state_len=state_len,interpret=interpret)
    def merge(a,b,out):
        out[...] = jnp.where(jnp.arange(128)[None]==0,
            ((a[0,0]!=0)|(b[0,0]!=0)).astype(jnp.uint32),jnp.uint32(0))
    merge_call=pl.pallas_call(merge,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_materialization_round_error')
    def local_count(counts,out):
        rank=jnp.int32(0) if mesh.size==1 else lax.axis_index('core')
        # One selected uint32 value; signed reduction retains all its bits.
        out[0]=jnp.sum(jnp.where(jnp.arange(mesh.size)==rank,counts[...],
            jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
    local_count_call=pl.pallas_call(local_count,
        out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
        interpret=interpret,name='beam_materialization_local_count')
    def call(state,parents,generators,prepared,target_counts,chunk):
        if target_counts.shape!=(mesh.size,) or target_counts.dtype!=jnp.uint32:
            raise ValueError('invalid materialization target counts')
        if generators.shape[0]!=move_count:
            raise ValueError('materialization move count mismatch')
        requests,ri,history,hi,plan_error=prepared
        snapshots,counts,received_history,history_status,error=delivery(
            requests,ri,history,hi,merge_call(state.error,plan_error),chunk)
        target_count=local_count_call(target_counts)
        history,hm,hc,error=consume_history(state.history,state.history_marks,
            state.history_control,received_history,history_status,target_count,error)
        grouped,intervals,error=pallas_final_response_plan(parents,generators,
            snapshots,counts,error,target_counts,state_len=state_len,
            world_size=mesh.size,interpret=interpret)
        exchange=make_final_response_chunk_call(mesh,wire_width=parents.shape[1],interpret=interpret)
        frontier,rm,rc=state.frontier,state.response_marks,state.response_control
        for subepoch in range(mesh.size):
            wire,status=exchange(grouped,intervals,error,jnp.array([subepoch],jnp.uint32))
            frontier,rm,rc,error=consume_response(frontier,rm,rc,wire,status,target_count,error)
        return FinalMaterializationState(frontier,history,rm,rc,hm,hc,error)
    return call
