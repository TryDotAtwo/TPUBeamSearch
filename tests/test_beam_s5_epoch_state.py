import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('jobs,force,expected',[(0,0,0),(2,0,0),(3,0,1),(0,1,1)])
def test_request_uses_completed_jobs_or_force(jobs,force,expected):
    from tpu_beam_search.beam_s5_epoch_state import pallas_s5_local_request
    state = np.zeros((4,128),np.uint32)
    state[0,0] = jobs
    got = pallas_s5_local_request(jnp.asarray(state),jnp.array([force],jnp.uint32),
        period=3,interpret=True)
    want = np.zeros((1,128),np.uint32)
    want[0,0] = expected
    np.testing.assert_array_equal(got,want)


@pytest.mark.parametrize('published',[0,1])
def test_epoch_only_resets_after_publication(published):
    from tpu_beam_search.beam_s5_epoch_state import pallas_s5_complete_epoch
    state = np.full((4,128),71,np.uint32)
    state[:,0] = [2,9,1,1]  # jobs, updates, local/global request.
    want = state.copy()
    if published:
        want[:,0] = [0,10,0,0]
    got = pallas_s5_complete_epoch(jnp.asarray(state),jnp.array([published],jnp.uint32),interpret=True)
    np.testing.assert_array_equal(got,want)
