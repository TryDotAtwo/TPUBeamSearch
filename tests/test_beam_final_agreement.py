from types import SimpleNamespace
import jax.numpy as jnp
import pytest
import jax
import numpy as np


@pytest.mark.parametrize('case',['valid','duplicate_across_chunks','late_error'])
def test_decoded_whole_response_coverage_and_late_error(case):
    from tpu_beam_search.beam_final_response import pallas_unpack_response
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    # Already assembled private responses: epoch0 has128 records, epoch1 has1.
    # Encode by literal LE bytes, independently of the production pack helper.
    wire = np.full((256,128),255,np.uint8)
    for slot in range(129):
        target = 0 if case == 'duplicate_across_chunks' and slot == 128 else slot
        wire[slot,120:124] = list(target.to_bytes(4,'little'))
    _,targets = pallas_unpack_response(jnp.asarray(wire),state_len=120,interpret=True)
    valid = jnp.asarray((np.arange(256)<129).astype(np.uint32)[None,:])
    prior = jnp.zeros((1,128),jnp.uint32).at[0,0].set(int(case == 'late_error'))
    common,summary = make_final_coverage_agreement(
        SimpleNamespace(size=1),interpret=True)(targets,valid,jnp.array([129],jnp.uint32),prior)
    assert int(common[0,0]) == int(case != 'valid')
    assert (int(summary[0,0]) != 0) == (case == 'duplicate_across_chunks')
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    candidate = np.full((129,128),37,np.uint8)
    actual,scatter_error = pallas_scatter_final_responses(jnp.asarray(candidate.copy()),
        jnp.asarray(wire),jnp.array([129],jnp.uint32),state_len=120,
        prior_error=common,interpret=True)
    expected = candidate.copy()
    if case == 'valid':
        expected[:,:120] = 255
        expected[:,120:] = 0
    np.testing.assert_array_equal(actual,expected)
    assert (int(scatter_error[0,0]) != 0) == (case != 'valid')


@pytest.mark.parametrize('bad', [False,True])
def test_local_coverage_flows_into_common_error(bad):
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    call = make_final_coverage_agreement(SimpleNamespace(size=1),interpret=True)
    targets = jnp.arange(128,dtype=jnp.uint32)[None,:]
    if bad: targets = targets.at[0,1].set(0)
    common, summary = call(targets,jnp.ones((1,128),jnp.uint32),jnp.array([128],jnp.uint32))
    assert int(common[0,0]) == int(bad)
    assert (int(summary[0,0]) != 0) == bad


def test_eight_rank_coverage_agreement_traces():
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    call = make_final_coverage_agreement(SimpleNamespace(size=8))
    shapes = (jax.ShapeDtypeStruct((1,256),jnp.uint32),
              jax.ShapeDtypeStruct((1,256),jnp.uint32),
              jax.ShapeDtypeStruct((1,),jnp.uint32))
    traced = jax.make_jaxpr(call,axis_env=[('core',8)])(*shapes)
    assert [x.aval.shape for x in traced.jaxpr.outvars] == [(1,128),(2,128)]


def test_prior_transport_error_is_not_hidden_by_valid_coverage():
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    call = make_final_coverage_agreement(SimpleNamespace(size=1),interpret=True)
    targets = jnp.arange(128,dtype=jnp.uint32)[None,:]
    prior = jnp.zeros((1,128),jnp.uint32).at[0,0].set(0x80000000)
    common, summary = call(targets,jnp.ones_like(targets),jnp.array([128],jnp.uint32),prior)
    assert int(common[0,0]) == 1
    assert int(summary[0,0]) == 0
