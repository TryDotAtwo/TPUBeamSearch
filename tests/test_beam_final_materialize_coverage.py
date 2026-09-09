"""Local component integration; not remote transport or publication evidence."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('case', ['valid', 'duplicate', 'bad_parent', 'late_error'])
def test_materialized_wire_requires_coverage_and_common_error_before_scatter(case):
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    from tpu_beam_search.beam_final_response import pallas_unpack_response
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses

    parents = np.zeros((2,128),np.uint8)
    parents[0,:120] = np.arange(120,dtype=np.uint8)
    parents[1,:120] = np.arange(119,-1,-1,dtype=np.uint8)
    generators = np.arange(128,dtype=np.int32)[None,:]
    requests = np.full((4,128),0xffffffff,np.uint32)
    requests[:,:2] = 0
    requests[0,:2] = [1,0]
    requests[2,:2] = [1,0] if case != 'duplicate' else [0,0]
    if case == 'bad_parent':
        requests[1,1] = 1
    count = jnp.array([2],jnp.uint32)
    device_parents = jnp.asarray(parents.copy())
    wire,material_error = pallas_materialize_final(
        device_parents,jnp.asarray(generators),jnp.asarray(requests),count,count,
        state_len=120,return_counts=count,interpret=True)
    _,targets = pallas_unpack_response(wire,state_len=120,interpret=True)
    prior = material_error[:1,:]
    if case == 'late_error':
        prior = prior.at[0,0].set(1)
    valid = jnp.asarray((np.arange(128)<2).astype(np.uint32)[None,:])
    common,_ = make_final_coverage_agreement(SimpleNamespace(size=1),interpret=True)(
        targets,valid,count,prior)
    candidate = np.full((2,128),37,np.uint8)
    actual,error = pallas_scatter_final_responses(jnp.asarray(candidate.copy()),
        wire,count,state_len=120,prior_error=common,interpret=True)
    expected = candidate.copy()
    if case == 'valid':
        expected[:,:120] = parents[:,:120]
        expected[:,120:] = 0
    np.testing.assert_array_equal(actual,expected)
    assert int(common[0,0]) == int(case != 'valid')
    assert int(error[0,0]) == int(case != 'valid')
    # Source frontier is read-only throughout this local chain.
    np.testing.assert_array_equal(device_parents,parents)
