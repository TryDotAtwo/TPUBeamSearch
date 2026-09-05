import importlib.util
import numpy as np
import jax.numpy as jnp


def test_global_histogram_sum_preserves_high_words_and_cross_rank_carries():
    assert importlib.util.find_spec('tpu_beam_search.beam_histogram_pair_sum') is not None
    from tpu_beam_search.beam_histogram_pair_sum import pallas_sum_histogram_pairs
    rng = np.random.default_rng(602)
    values = rng.integers(0,1<<59,(8,256),dtype=np.uint64)
    values[:,0] = np.uint64(0xffffffff)
    values[:,129] = np.uint64(0x100000001)
    packed = np.empty((16,256),np.uint32)
    packed[0::2] = values.astype(np.uint32)
    packed[1::2] = (values >> np.uint64(32)).astype(np.uint32)
    summed = values.sum(axis=0,dtype=np.uint64)
    expected = np.stack((summed.astype(np.uint32),(summed >> np.uint64(32)).astype(np.uint32)))
    actual = pallas_sum_histogram_pairs(jnp.asarray(packed),interpret=True)
    np.testing.assert_array_equal(actual,expected)
    assert tuple(expected[:,0]) == (0xfffffff8,7)
    assert tuple(expected[:,129]) == (8,8)
