"""Checked response consumption into a private compact frontier."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_response import pallas_unpack_response
from .beam_final_streaming_coverage import pallas_mark_final_targets
from .beam_final_scatter import pallas_scatter_compact_final_responses
from .beam_s5_request import make_s5_request_call


def make_final_response_consumer(mesh,*,state_len,interpret=False):
    """All ranks consume the same epochs, including empty/error epochs.

    Merge receive/prior errors into sticky coverage, then mark target identities
    before any frontier store. Agree mark errors collectively; any rejected
    rank blocks the whole epoch's stores. Agree scatter errors afterwards too.
    Returns private frontier, marks, coverage control and common error.
    A failed mark batch may change PRIVATE marks; never retry/publish that
    depth. Old published frontier must not be supplied or donated here.
    History coverage, final mark scan and publication are separate obligations.
    This serialized implementation does not establish compute/DMA overlap.
    """
    agree=make_s5_request_call(mesh,interpret=interpret)
    def merge(c,r,e,out):
        out[0,:]=c[0,:]
        bad=(c[1,0]!=0)|(r[1,0]!=0)|(e[0,0]!=0)
        out[1,:]=jnp.where(jnp.arange(128)==0,bad.astype(jnp.uint32),jnp.uint32(0))
    merge_call=pl.pallas_call(merge,out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
        interpret=interpret,name='beam_response_consumer_error')
    def scatter_flag(e,s,out):
        bad=(e[0,0]!=0)|(s[0,0]!=0)
        out[...] = jnp.where(jnp.arange(128)[None]==0,bad.astype(jnp.uint32),jnp.uint32(0))
    scatter_flag_call=pl.pallas_call(scatter_flag,
        out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_response_consumer_scatter_error')
    def call(frontier,seen,coverage,wire,status,target_count,prior_error):
        if (status.shape!=(2,128) or coverage.shape!=(2,128)
                or prior_error.shape!=(1,128)
                or any(x.dtype!=jnp.uint32 for x in (status,coverage,prior_error))):
            raise ValueError('invalid response consumer controls')
        coverage=merge_call(coverage,status,prior_error)
        _,targets=pallas_unpack_response(wire,state_len=state_len,interpret=interpret)
        seen,coverage=pallas_mark_final_targets(seen,coverage,targets,status[0,:1],
            target_count,interpret=interpret)
        common=agree(coverage[1:2])
        frontier,scatter_errors=pallas_scatter_compact_final_responses(frontier,wire,
            status[0,:1],state_len=state_len,interpret=interpret,prior_error=common)
        common=agree(scatter_flag_call(common,scatter_errors))
        coverage=merge_call(coverage,status,common)
        return frontier,seen,coverage,common
    return call
