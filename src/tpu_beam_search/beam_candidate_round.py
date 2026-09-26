"""Serialized S2/solved/S3/S4/S5 integration with supplied score keys.

Still bounded to 128 candidates by the transport gate; not a full depth,
inference integration, final publication, or scratch-lifetime implementation.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_runner import StreamRoundState, make_stream_round_call
from .beam_stream2_batch import pallas_stream2_batch
from .beam_solved_service import make_solved_batch_service


class CandidateRoundState(NamedTuple):
    streams: StreamRoundState
    solved: jax.Array
    solved_control: jax.Array
    error: jax.Array
    stop: jax.Array


def make_candidate_round_call(mesh, *, bins, period, clean_ready_threshold,
                              dirty_trigger, stop_on_found, interpret=False):
    """Uniform collective schedule, even when no new parents are admitted.

    A stop applies to the next batch: a batch already admitted finishes its
    candidate/solution accounting. Fatal errors instead gate S3 immediately.
    Same-score replay inputs are uint32 keys, not actual model inference.
    K1/K2 optional arguments use the existing validated table contracts.
    """
    solved = make_solved_batch_service(mesh,local_rank=0 if mesh.size == 1 else None,
        stop_on_found=stop_on_found,interpret=interpret)
    streams = make_stream_round_call(mesh,bins=bins,period=period,
        clean_ready_threshold=clean_ready_threshold,dirty_trigger=dirty_trigger,
        interpret=interpret)

    def call(state,parents,generators,central,zobrist,parent_count,scores,
             parent_base,payload_base,depth,beam,force,neutral,*,
             k1_table=None,bucket_count=0,suffix_words=None,suffix_count=0):
        if parents.shape[0]*generators.shape[0] > 128:
            raise ValueError('current integrated transport supports at most128 candidates')
        if any(x.shape != (1,128) or x.dtype != jnp.uint32 for x in (state.error,state.stop)):
            raise ValueError('round error/stop must be uint32 [1,128]')
        shards = state.streams.controls.shape[0]
        def admission(c,e,s,n,out,prior):
            fatal = e[0,0] != 0
            for shard in range(shards):
                fatal |= c[shard,7,0] != 0
            out[0] = jnp.where(fatal | (s[0,0] != 0),jnp.uint32(0),n[0])
            prior[...] = jnp.where(jnp.arange(128)[None] == 0,
                                   fatal.astype(jnp.uint32),jnp.uint32(0))
        count,prior = pl.pallas_call(admission,
            out_shape=(jax.ShapeDtypeStruct((1,),jnp.uint32),
                       jax.ShapeDtypeStruct((1,128),jnp.uint32)),
            interpret=interpret,name='beam_candidate_round_admission')(
                state.streams.controls,state.error,state.stop,parent_count)
        batch = pallas_stream2_batch(parents,generators,central,zobrist,count,
            scores,parent_base,payload_base,k1_table=k1_table,bucket_count=bucket_count,
            suffix_words=suffix_words,suffix_count=suffix_count,interpret=interpret)
        arena,ctl,solved_error,solved_stop = solved(
            state.solved,state.solved_control,batch,depth,prior)
        def fatal_control(c,e,out):
            out[...] = c[...]
            for shard in range(shards):
                out[shard,7,:] = jnp.where(jnp.arange(128) == 0,
                    c[shard,7,:] | e[0,0],c[shard,7,:])
        controls = pl.pallas_call(fatal_control,
            out_shape=jax.ShapeDtypeStruct(state.streams.controls.shape,jnp.uint32),
            interpret=interpret,name='beam_candidate_round_common_fatal')(
                state.streams.controls,solved_error)
        stream,error,jobs = streams(state.streams._replace(controls=controls),
            batch.words,batch.payload,batch.count,beam,force,neutral)
        def sticky(e,se,s,ss,err,stop):
            failed = (e[0,0] != 0) | (se[0,0] != 0)
            lanes = jnp.arange(128)[None] == 0
            err[...] = jnp.where(lanes,failed.astype(jnp.uint32),jnp.uint32(0))
            stop[...] = jnp.where(lanes,(failed | (s[0,0] != 0) | (ss[0,0] != 0)).astype(jnp.uint32),jnp.uint32(0))
        error,stop = pl.pallas_call(sticky,
            out_shape=(jax.ShapeDtypeStruct((1,128),jnp.uint32),)*2,
            interpret=interpret,name='beam_candidate_round_sticky_decision')(
                error,solved_error,state.stop,solved_stop)
        return CandidateRoundState(stream,arena,ctl,error,stop),jobs
    return call
