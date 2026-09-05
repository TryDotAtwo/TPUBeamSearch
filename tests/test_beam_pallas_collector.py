import numpy as np
import jax.numpy as jnp
import pytest
from tpu_beam_search import beam_collector as module


@pytest.mark.parametrize('used,busy,amount,target', [
    ((129,0),(0,0),127,0), ((250,0),(0,0),128,1),
    ((0,129),(1,0),127,1), ((250,250),(0,0),128,None),
    ((256,256),(1,1),0,None)])
def test_pallas_append_preserves_sibling_and_existing_records(used,busy,amount,target):
    assert hasattr(module,'pallas_collector_append')
    a = np.arange(8*256,dtype=np.uint32).reshape(8,256)
    b = a + np.uint32(10000)
    incoming = np.arange(8*128,dtype=np.uint32).reshape(8,128)+np.uint32(0x80000000)
    # rows: clean A/B, dirty A/B, busy A/B, current, sticky fatal; lane zero.
    control = np.zeros((8,128),np.uint32)
    control[:2,0] = used
    control[4:6,0] = busy
    count = np.zeros((1,128),np.uint32)
    count[0,0] = amount
    actual = module.pallas_collector_append(*map(jnp.asarray,(a,b,incoming,control,count)),interpret=True)
    expected = [a.copy(),b.copy(),control.copy()]
    if target is not None:
        expected[target][:,used[target]:used[target]+amount] = incoming[:,:amount]
        expected[2][2+target,0] = amount
        expected[2][6,0] = target
    elif amount:
        expected[2][7,0] = 1
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_array_equal(got,want)


def test_repeated_arrivals_fill_siblings_then_sticky_fatal_blocks_writes():
    a = jnp.zeros((8,256),jnp.uint32)
    b = jnp.zeros_like(a)
    control = jnp.zeros((8,128),jnp.uint32)
    count = jnp.zeros((1,128),jnp.uint32).at[0,0].set(128)
    for value in range(1,5):
        incoming = jnp.full((8,128),value,jnp.uint32)
        a,b,control = module.pallas_collector_append(a,b,incoming,control,count,interpret=True)
    np.testing.assert_array_equal(a,np.tile(np.repeat([1,2],128),(8,1)))
    np.testing.assert_array_equal(b,np.tile(np.repeat([3,4],128),(8,1)))
    np.testing.assert_array_equal(control[2:4,0],[256,256])
    before = (np.asarray(a).copy(),np.asarray(b).copy())
    a,b,control = module.pallas_collector_append(a,b,incoming,control,count,interpret=True)
    assert int(control[7,0]) == 1
    # Even if a caller later frees space, fatal is sticky until explicit reset.
    control = control.at[2:4,0].set(0)
    a,b,after = module.pallas_collector_append(a,b,incoming,control,count,interpret=True)
    np.testing.assert_array_equal(a,before[0])
    np.testing.assert_array_equal(b,before[1])
    np.testing.assert_array_equal(after,control)
