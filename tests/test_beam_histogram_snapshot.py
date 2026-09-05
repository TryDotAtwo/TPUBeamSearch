import importlib.util
import numpy as np
import jax.numpy as jnp


def test_committed_histogram_snapshot_selects_active_and_preserves_uint64_carry():
    assert importlib.util.find_spec('tpu_beam_search.beam_histogram_snapshot') is not None
    from tpu_beam_search.beam_histogram_snapshot import pallas_sum_committed_histograms
    a,b = np.zeros((8,256),np.uint32),np.zeros((8,256),np.uint32)
    active = np.zeros((1,128),np.uint32)
    active[0,:8] = [0,1,0,1,0,1,0,1]
    for rank in range(8):
        chosen,other = (a,b) if rank%2 == 0 else (b,a)
        chosen[rank,0] = 0xffffffff
        chosen[rank,128] = rank+1
        other[rank] = 12345
    result = pallas_sum_committed_histograms(*map(jnp.asarray,(a,b,active)),interpret=True)
    expected = np.zeros((2,256),np.uint32)
    expected[:,0] = (0xfffffff8,7)
    expected[0,128] = 36
    np.testing.assert_array_equal(result,expected)
