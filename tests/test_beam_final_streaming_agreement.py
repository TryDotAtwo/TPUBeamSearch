from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('failure',[None,'response_hole','history_hole','history_error','prior'])
def test_final_agreement_requires_both_complete_streams(failure):
    from tpu_beam_search.beam_final_streaming_agreement import make_final_streaming_agreement
    marks=jnp.zeros((1,128),jnp.uint32).at[0,:3].set(1)
    response,history=marks,marks
    control=jnp.zeros((2,128),jnp.uint32).at[0,0].set(3)
    rc,hc=control,control
    prior=jnp.zeros((1,128),jnp.uint32)
    if failure=='response_hole': response=response.at[0,1].set(0)
    if failure=='history_hole': history=history.at[0,1].set(0)
    if failure=='history_error': hc=hc.at[1,0].set(1)
    if failure=='prior': prior=prior.at[0,0].set(0x80000000)
    common,re,he=map(np.asarray,make_final_streaming_agreement(SimpleNamespace(size=1),interpret=True)(
        response,rc,history,hc,jnp.array([3],jnp.uint32),prior))
    assert common[0,0]==int(failure is not None)
    assert re[0,0]==int(failure=='response_hole')
    assert he[0,0]==int(failure in ('history_hole','history_error'))
    assert not common[0,1:].any()
