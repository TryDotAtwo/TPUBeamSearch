import importlib.util
import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('beam_lo,beam_hi,old,initialized,want,new_init',[
    (1,1,0xffffffff,0,130,1),(1,1,7,1,7,1),
    (3,1,0xffffffff,0,0xffffffff,0),(3,1,7,1,7,1),
])
def test_periodic_threshold_carry_and_no_relaxation(beam_lo,beam_hi,old,initialized,want,new_init):
    assert importlib.util.find_spec('tpu_beam_search.beam_threshold') is not None
    from tpu_beam_search.beam_threshold import pallas_periodic_threshold
    hist = np.zeros((2,256),np.uint32)
    hist[0,0],hist[0,130] = 0xffffffff,2
    beam,prior = np.zeros((2,128),np.uint32),np.zeros((2,128),np.uint32)
    beam[:,0],prior[:,0] = (beam_lo,beam_hi),(old,initialized)
    actual = pallas_periodic_threshold(*map(jnp.asarray,(hist,beam,prior)),bins=131,interpret=True)
    expected = np.zeros((2,128),np.uint32)
    expected[:,0] = (want,new_init)
    np.testing.assert_array_equal(actual,expected)
