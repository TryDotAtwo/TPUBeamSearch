import importlib.util
import numpy as np
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
import pytest


@pytest.mark.parametrize('active',[0,1])
@pytest.mark.parametrize('old,initialized,value,valid,want,want_init',[
    (0xffffffff,0,17,1,17,1),(12,1,17,1,12,1),
    (12,1,7,1,7,1),(12,1,0xffffffff,0,12,1),
    (0xffffffff,0,0xffffffff,0,0xffffffff,0)])
def test_periodic_publication_preserves_active_slot_and_never_relaxes(
        active,old,initialized,value,valid,want,want_init):
    assert importlib.util.find_spec('tpu_beam_search.beam_threshold_publish') is not None
    from tpu_beam_search.beam_threshold_publish import pallas_publish_periodic_threshold
    a,b = np.full((2,128),31,np.uint32),np.full((2,128),47,np.uint32)
    (a if active == 0 else b)[:,0] = (old,initialized)
    c = np.full((1,128),53,np.uint32)
    c[0,0] = active
    candidate = np.zeros((2,128),np.uint32)
    candidate[:,0] = (value,valid)
    result = pallas_publish_periodic_threshold(*map(jnp.asarray,(a,b,c,candidate)),
        interpret=pltpu.InterpretParams(detect_races=True))
    expected = [a.copy(),b.copy(),c.copy()]
    expected[active^1][:,0] = (want,want_init)
    expected[2][0,0] = active^1
    for got,ref in zip(result,expected,strict=True):
        np.testing.assert_array_equal(got,ref)
