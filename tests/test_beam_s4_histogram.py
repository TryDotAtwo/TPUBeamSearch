import importlib.util
import numpy as np
import jax.numpy as jnp
import pytest


def test_histogram_masks_count_and_out_of_domain_scores_without_clamping():
    assert importlib.util.find_spec('tpu_beam_search.beam_s4_histogram') is not None
    from tpu_beam_search.beam_s4_histogram import pallas_score_histogram
    words = np.zeros((8,256),np.uint32)
    words[6] = 5
    words[6,:7] = [0,0,128,256,257,0xffffffff,128]
    actual = pallas_score_histogram(jnp.asarray(words),jnp.array([7],jnp.uint32),
                                  bins=257,interpret=True)
    expected = np.zeros((1,384),np.uint32)
    expected[0,0],expected[0,128],expected[0,256] = 2,2,1
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('valid,want',[(0,0),(256,256)])
def test_histogram_empty_and_one_run_spanning_every_tile(valid,want):
    from tpu_beam_search.beam_s4_histogram import pallas_score_histogram
    words = np.zeros((8,256),np.uint32)
    words[6] = 127
    actual = pallas_score_histogram(jnp.asarray(words),jnp.array([valid],jnp.uint32),
                                  bins=128,interpret=True)
    expected = np.zeros((1,128),np.uint32)
    expected[0,127] = want
    np.testing.assert_array_equal(actual,expected)
