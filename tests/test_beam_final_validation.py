import numpy as np
import jax.numpy as jnp


def test_final_validation_uses_full_parent_width_and_masks_padding():
    from tpu_beam_search.beam_final_validation import pallas_validate_final_requests
    requests = np.zeros((4,128),np.uint32)
    requests[2] = np.arange(128,dtype=np.uint32)
    requests[0,:5] = [3,4,0,0,0]
    requests[1,:5] = [1,1,2,0,0]
    requests[2,3] = 9
    requests[3,4] = 24<<16
    reasons = pallas_validate_final_requests(jnp.asarray(requests),
        jnp.array([5],jnp.uint32),jnp.array([4,1],jnp.uint32),
        jnp.array([5],jnp.uint32),move_count=24,require_local_slot=True,interpret=True)
    want = np.zeros((1,128),np.uint32)
    want[0,:5] = [0,1,1,2|8,4]
    np.testing.assert_array_equal(reasons,want)
