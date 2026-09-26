"""Serialized metadata-stream round for the future full beam depth caller.

Runs real S3/collector/(multi-rank RDMA)/S4/S5, but not S1/S2, final selection,
frontier/history publication, solved handling or scratch phase transitions.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_remote_exchange import make_stream3_collect_call
from .beam_s4_s5_service import make_s4_s5_service_call, _pack_histograms


class StreamRoundState(NamedTuple):
    """A/B [S,8,C], controls [S,8,128], histograms [2*S,H], uint32.

    Hist-active and threshold-active are [1,128]; epoch is [4,128]; threshold
    slots are [2,128]. This metadata state is not the public frontier/history.
    """
    a: jax.Array
    b: jax.Array
    controls: jax.Array
    hist_a: jax.Array
    hist_b: jax.Array
    hist_active: jax.Array
    epoch: jax.Array
    threshold_a: jax.Array
    threshold_b: jax.Array
    threshold_active: jax.Array


def _stack_tiles(arrays, *, interpret):
    """Bounded tile copy into an array; not a claim of zero-copy state views."""
    planes,width = arrays[0].shape
    def kernel(*refs):
        for index,ref in enumerate(refs[:-1]):
            refs[-1][index,:,:] = ref[...]
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((len(arrays),planes,width),jnp.uint32),
        in_specs=(pl.BlockSpec((planes,128),lambda tile:(0,tile)),)*len(arrays),
        out_specs=pl.BlockSpec((len(arrays),planes,128),lambda tile:(0,0,tile)),
        grid=(width//128,),interpret=interpret,name='beam_stream_round_stack')(*arrays)


def make_stream_round_call(mesh, *, bins,period,clean_ready_threshold,dirty_trigger,
                            force_dirty=False,force_clean=False,interpret=False):
    """One fixed-schedule round; every rank calls even with count zero/fatal.

    Local inputs are at most 128 candidate records; the existing transport
    retains per-peer snapshots. Caller provides valid source metadata, explicit
    count, aligned resident arrays and complete committed histograms. Force
    controls are for S4 drain; the separate force argument requests S5 refresh.
    A pre-existing local fatal admits zero new records but still drains the
    common transport/collective sequence. The caller stops subsequent admission
    once the returned common error is observed on device.

    Interpretation is supported only for the true single-rank local path, not
    by silently replacing RDMA. Multi-rank mode uses the existing physical
    exchange. No host candidate-count/decision extraction occurs here.
    """
    collect = make_stream3_collect_call(mesh,interpret=interpret)
    service = make_s4_s5_service_call(mesh,bins=bins,period=period,
        clean_ready_threshold=clean_ready_threshold,dirty_trigger=dirty_trigger,
        force_dirty=force_dirty,force_clean=force_clean,interpret=interpret)
    def call(state,words,payload,count,beam,force,neutral):
        shards = state.a.shape[0]
        width = ((bins+127)//128)*128
        if (not 1 <= shards <= 64 or state.a.ndim != 3 or state.a.shape[1] != 8
                or state.b.shape != state.a.shape or state.controls.shape != (shards,8,128)
                or state.hist_a.shape != (2*shards,width) or state.hist_b.shape != state.hist_a.shape
                or state.hist_active.shape != (1,128)):
            raise ValueError('invalid stream round storage geometry')
        slots = (state.threshold_a,state.threshold_b,state.threshold_active)
        def capture(a,b,active,out):
            selected = jnp.where(active[0,0] == 0,a[...],b[...])
            out[0] = jnp.where(selected[1,0] != 0,selected[0,0],jnp.uint32(0xffffffff))
        threshold = pl.pallas_call(capture,out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
            interpret=interpret,name='beam_stream_round_threshold')(*slots)
        def admission(c,n,out):
            fatal = jnp.uint32(0)
            for shard in range(shards):
                fatal |= (c[shard,7,0] != 0).astype(jnp.uint32)
            out[0] = jnp.where(fatal != 0,jnp.uint32(0),n[0])
        admitted = pl.pallas_call(admission,out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
            interpret=interpret,name='beam_stream_round_admission')(state.controls,count)
        a,b,controls,_ = collect(state.a,state.b,state.controls,words,payload,admitted,threshold,neutral)
        pairs = []
        for shard in range(shards):
            def versions(ref,out):
                lanes = jnp.arange(128)[None]
                out[...] = jnp.where(lanes == 0,ref[0,2*shard],
                                     jnp.where(lanes == 1,ref[0,2*shard+1],jnp.uint32(0)))
            active = pl.pallas_call(versions,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
                interpret=interpret,name='beam_stream_round_histogram_version')(state.hist_active)
            start = 2*shard
            pairs.append((a[shard],b[shard],controls[shard],
                (state.hist_a[start:start+1],state.hist_a[start+1:start+2]),
                (state.hist_b[start:start+1],state.hist_b[start+1:start+2]),active))
        pairs,epoch,slots,error,jobs = service(tuple(pairs),state.epoch,slots,beam,force)
        hist_a,hist_b,active = _pack_histograms(pairs,interpret=interpret)
        next_state = StreamRoundState(
            _stack_tiles(tuple(p[0] for p in pairs),interpret=interpret),
            _stack_tiles(tuple(p[1] for p in pairs),interpret=interpret),
            _stack_tiles(tuple(p[2] for p in pairs),interpret=interpret),
            hist_a,hist_b,active,epoch,*slots)
        return next_state,error,_stack_tiles(jobs,interpret=interpret)
    return call
