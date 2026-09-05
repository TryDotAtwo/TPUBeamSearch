import importlib.util
import numpy as np
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
import pytest


@pytest.mark.parametrize('active',[0,1])
def test_histogram_dma_commit_preserves_active_buffer_and_releases_after_copy(active):
    assert importlib.util.find_spec('tpu_beam_search.beam_histogram_commit') is not None
    from tpu_beam_search.beam_histogram_commit import pallas_commit_histogram
    a = np.full((1,384),11,np.uint32)
    b = np.full((1,384),22,np.uint32)
    new = np.arange(384,dtype=np.uint32)[None]
    control = np.zeros((2,128),np.uint32)
    control[:,0] = (active,1)
    got_a,got_b,got_c = pallas_commit_histogram(*map(jnp.asarray,(a,b,control,new)),
        interpret=pltpu.InterpretParams(detect_races=True))
    np.testing.assert_array_equal(got_a,a if active == 0 else new)
    np.testing.assert_array_equal(got_b,new if active == 0 else b)
    expected = control.copy()
    expected[:,0] = (active^1,0)
    np.testing.assert_array_equal(got_c,expected)
