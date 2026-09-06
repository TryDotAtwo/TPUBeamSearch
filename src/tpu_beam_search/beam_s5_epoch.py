"""Serialized S5 composition; physical distributed epoch acceptance pending."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_s5_request import make_s5_request_call
from .beam_s5_histogram_exchange import make_s5_histogram_call
from .beam_histogram_snapshot import pallas_sum_committed_histograms
from .beam_threshold import pallas_periodic_threshold
from .beam_threshold_publish import pallas_publish_periodic_threshold
from .beam_s5_epoch_state import pallas_s5_local_request,pallas_s5_complete_epoch


def make_s5_epoch_call(mesh,*,bins,period,interpret=False,explicit_hbm_output=False):
    """All ranks call under a common core mesh; never branch on local request.

    Caller drains S4 writers and threshold readers before entry, freezes selected
    histograms, owns counters, and keeps buffers alive through completion.
    JAX supplies uniform conditional orchestration, Pallas supplies arithmetic
    and DMA. This is not concurrent reader protection or overlap.
    """
    request_call = make_s5_request_call(mesh,interpret=interpret)
    width = ((bins+127)//128)*128
    histogram_call = make_s5_histogram_call(mesh,width=width,interpret=interpret,
        explicit_hbm_output=explicit_hbm_output)

    def select(a,b,active,out):
        out[...] = jnp.where(active[0,0] == 0,a[...],b[...])
    select_call = pl.pallas_call(select,
        out_shape=jax.ShapeDtypeStruct((2,128),jnp.uint32),
        interpret=interpret,name='beam_s5_capture_prior')

    def completed(active,out):
        # Consumes the actual output of threshold publication; nonzero for
        # either valid active slot. Not a cross-rank completion barrier.
        out[0] = active[0,0]+jnp.uint32(1)
    complete_call = pl.pallas_call(completed,
        out_shape=jax.ShapeDtypeStruct((1,),jnp.uint32),
        interpret=interpret,name='beam_s5_publication_dependency')

    def call(hist_a,hist_b,hist_active,a,b,active,beam,state,force):
        request = pallas_s5_local_request(state,force,period=period,interpret=interpret)
        common = request_call(request)
        def update(_):
            snapshot = pallas_sum_committed_histograms(hist_a,hist_b,hist_active,interpret=interpret)
            total = histogram_call(snapshot)
            prior = select_call(a,b,active)
            candidate = pallas_periodic_threshold(total,beam,prior,bins=bins,interpret=interpret)
            na,nb,nactive = pallas_publish_periodic_threshold(a,b,active,candidate,interpret=interpret)
            nstate = pallas_s5_complete_epoch(state,complete_call(nactive),interpret=interpret)
            return na,nb,nactive,nstate
        return jax.lax.cond(common[0,0] != 0,update,lambda _:(a,b,active,state),None)
    return call
