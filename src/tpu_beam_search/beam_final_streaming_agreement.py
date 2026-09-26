"""Collective final decision over both private streaming coverage maps."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_streaming_coverage import pallas_finish_final_targets
from .beam_s5_request import make_s5_request_call


def make_final_streaming_agreement(mesh,*,interpret=False):
    """Every rank calls after all response/history epochs, even on error.

    Checks both mark maps independently, including counts and sticky errors.
    Returns common error and the two local errors. This decision is NOT a
    publication event or proof that unrelated DMA has drained: actual frontier
    and history consumer outputs must stay live and be awaited by publication.
    """
    agree=make_s5_request_call(mesh,interpret=interpret)
    def merge(r,h,p,out):
        bad=(r[0,0]!=0)|(h[0,0]!=0)|(p[0,0]!=0)
        out[...] = jnp.where(jnp.arange(128)[None]==0,bad.astype(jnp.uint32),jnp.uint32(0))
    flag=pl.pallas_call(merge,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_streaming_agreement_flag')
    def call(response_marks,response_control,history_marks,history_control,target_count,prior_error):
        if prior_error.shape!=(1,128) or prior_error.dtype!=jnp.uint32:
            raise ValueError('invalid final streaming prior error')
        response=pallas_finish_final_targets(response_marks,response_control,target_count,interpret=interpret)
        history=pallas_finish_final_targets(history_marks,history_control,target_count,interpret=interpret)
        return agree(flag(response,history,prior_error)),response,history
    return call
