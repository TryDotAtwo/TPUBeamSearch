import importlib.util
import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('clean,dirty,busy,current,expected,write',[
    ((0,0),(64,0),(0,0),0,0,1),
    ((0,0),(64,64),(0,0),0,1,0),
    ((250,0),(0,0),(0,0),0,0,1),
    ((0,0),(63,0),(0,0),0,None,0),
    ((0,0),(64,64),(1,0),0,None,0),
    ((240,0),(0,64),(0,0),0,1,0),
    ((256,256),(0,0),(0,0),0,0,0),
    ((256,256),(0,0),(0,0),1,1,1),
    ((0,0),(64,64),(0,0),1,0,1),
    ((0,0),(64,64),(0,1),1,None,1),
    ((0,0),(80,64),(0,0),0,0,1),
])
def test_ready_claim_preserves_source_priority_and_sibling_exclusion(clean,dirty,busy,current,expected,write):
    assert importlib.util.find_spec('tpu_beam_search.beam_s4_ready') is not None
    from tpu_beam_search.beam_s4_ready import pallas_claim_ready
    controls = np.zeros((8,128),np.uint32)
    controls[:7,0] = (*clean,*dirty,*busy,current)
    actual,job = pallas_claim_ready(jnp.asarray(controls),capacity=256,
        clean_ready_threshold=240,dirty_trigger=64,interpret=True)
    want = controls.copy()
    if expected is not None:
        want[4+expected,0] = 1
        want[6,0] = write
    np.testing.assert_array_equal(actual,want)
    want_job = np.zeros((2,128),np.uint32)
    if expected is not None:
        want_job[:,0] = (1,expected)
    np.testing.assert_array_equal(job,want_job)


@pytest.mark.parametrize('clean,dirty,force_dirty,force_clean,enabled',[
    (0,1,True,False,1),(1,0,False,True,1),
    (0,1,False,True,0),(1,0,True,False,0),(0,0,True,True,0),
])
def test_force_drain_respects_dirty_vs_clean_and_never_claims_empty(clean,dirty,force_dirty,force_clean,enabled):
    from tpu_beam_search.beam_s4_ready import pallas_claim_ready
    c = np.zeros((8,128),np.uint32)
    c[0,0],c[2,0] = clean,dirty
    actual,job = pallas_claim_ready(jnp.asarray(c),capacity=256,
        clean_ready_threshold=240,dirty_trigger=64,force_dirty=force_dirty,
        force_clean=force_clean,interpret=True)
    want = c.copy()
    want[4,0],want[6,0] = enabled,enabled
    np.testing.assert_array_equal(actual,want)
    assert int(job[0,0]) == enabled


@pytest.mark.parametrize('capacity,shards,batch,want',[
    (256,3,128,202),(256,8,512,176),(256,1,512,0),
    (256,256,1,254),(0xffffffff,1,0xffffffff,0),
])
def test_ready_threshold_reserves_ceiling_average_plus_quarter(capacity,shards,batch,want):
    from tpu_beam_search import beam_s4_ready
    assert hasattr(beam_s4_ready,'clean_ready_threshold')
    assert beam_s4_ready.clean_ready_threshold(capacity=capacity,logical_shards=shards,
                                              stream3_batch=batch) == want
