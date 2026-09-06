import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('less,equal,beam',[(0,0,0),(3,9,7),(3,9,100),(9,3,7),(0xffffffff,9,1<<32)])
def test_exact_cap_and_invalid_threshold_gate(less,equal,beam):
    from tpu_beam_search.beam_final_cap import pallas_final_cap
    totals = np.zeros((4,128),np.uint32)
    target = np.zeros((2,128),np.uint32)
    for phase,value in enumerate((less,equal)):
        totals[2*phase,0],totals[2*phase+1,0] = value&0xffffffff,value>>32
    target[:,0] = beam&0xffffffff,beam>>32
    keep,error = pallas_final_cap(jnp.asarray(totals),jnp.asarray(target),interpret=True)
    count = min(beam,less+equal)
    expected = np.zeros((2,128),np.uint32)
    expected[:,0] = count&0xffffffff,count>>32
    np.testing.assert_array_equal(keep,expected)
    expected_error = np.zeros((1,128),np.uint32)
    expected_error[0,0] = less>count
    np.testing.assert_array_equal(error,expected_error)
