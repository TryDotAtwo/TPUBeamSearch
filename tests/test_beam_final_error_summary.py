import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('indices',[[],[0],[127,128,255]])
def test_final_error_summary_counts_and_selects_first_across_tiles(indices):
    from tpu_beam_search.beam_final_error_summary import pallas_final_error_summary
    reasons = np.zeros((1,256),np.uint32)
    reasons[0,indices] = 5
    got = pallas_final_error_summary(jnp.asarray(reasons),interpret=True)
    want = np.zeros((2,128),np.uint32)
    want[0,0] = len(indices)
    want[1,0] = indices[0] if indices else 0xffffffff
    np.testing.assert_array_equal(got,want)
