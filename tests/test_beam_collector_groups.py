"""Whole-group reservation must not become separate per-tile admissions."""
import numpy as np
import jax.numpy as jnp
import pytest
from tpu_beam_search import beam_collector as module


@pytest.mark.parametrize('used,amount,target', [((300,0),256,1),
                                               ((300,300),256,None),
                                               ((255,0),257,0)])
def test_group_reserves_one_sibling_before_any_tile_write(used,amount,target):
    assert hasattr(module,'pallas_collector_append_group')
    a = np.arange(8*512,dtype=np.uint32).reshape(8,512)
    b = a + np.uint32(10000)
    incoming = a + np.uint32(0x80000000)
    control = np.zeros((8,128),np.uint32)
    control[:2,0] = used
    count = np.zeros((1,128),np.uint32)
    count[0,0] = amount
    actual = module.pallas_collector_append_group(
        *map(jnp.asarray,(a,b,incoming,control,count)),interpret=True)
    expected = [a.copy(),b.copy(),control.copy()]
    if target is None:
        expected[2][7,0] = 1
    else:
        expected[target][:,used[target]:used[target]+amount] = incoming[:,:amount]
        expected[2][2+target,0] = amount
        expected[2][6,0] = target
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_array_equal(got,want)
