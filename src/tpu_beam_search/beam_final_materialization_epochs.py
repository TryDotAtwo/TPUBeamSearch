"""Uniform request/history epoch loop over private final materialization arenas."""
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from .beam_final_epoch_schedule import make_final_epoch_schedule
from .beam_final_materialization_round import make_final_materialization_round
from .beam_final_streaming_agreement import make_final_streaming_agreement


def make_final_materialization_epochs(mesh, *, state_len, move_count,
                                      request_capacity, history_capacity,
                                      interpret=False):
    """Drain both paired streams and check coverage before host publication.

    All ranks derive the same bounded epoch count and execute the same number
    of collective rounds. Outputs remain private. The caller must await these
    arrays and perform atomic publication using the common error result.
    """
    schedule = make_final_epoch_schedule(mesh,
        request_capacity=request_capacity, history_capacity=history_capacity,
        interpret=interpret)
    round_call = make_final_materialization_round(mesh, state_len=state_len,
        move_count=move_count, interpret=interpret)
    finish = make_final_streaming_agreement(mesh, interpret=interpret)
    def merge_prior(a,b,out):
        bad=(a[0,0]!=0)|(b[0,0]!=0)
        out[...] = jnp.where(jnp.arange(128)[None,:]==0,
                             bad.astype(jnp.uint32),jnp.uint32(0))
    merge_call=pl.pallas_call(merge_prior,
        out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_epochs_prior_error')

    def call(state, parents, generators, prepared, target_counts, local_target_count):
        requests, request_intervals, history, history_intervals, plan_error = prepared
        if requests.shape[1] != request_capacity or history.shape[1] != history_capacity:
            raise ValueError('prepared exchange capacity mismatch')
        if local_target_count.shape != (1,) or local_target_count.dtype != jnp.uint32:
            raise ValueError('local target count must be uint32 [1]')
        rounds, schedule_error = schedule(request_intervals, history_intervals,
                                          merge_call(state.error,plan_error))
        state = state._replace(error=schedule_error)

        def body(epoch, current):
            return round_call(current, parents, generators, prepared,
                              target_counts, jnp.asarray([epoch], jnp.uint32))

        state = lax.fori_loop(0, rounds[0].astype(jnp.int32), body, state)
        common, response_local, history_local = finish(
            state.response_marks, state.response_control,
            state.history_marks, state.history_control,
            local_target_count, state.error)
        return state._replace(error=common), response_local, history_local

    return call
