import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search import beam_collector as module


@pytest.mark.parametrize('late_overflow',[False,True])
def test_all_shard_preflight_gates_earlier_groups_on_late_overflow(late_overflow):
    assert hasattr(module,'pallas_collector_preflight')
    controls = np.zeros((2,8,128),np.uint32)
    controls[0,0,0] = 200  # group 128 must select B, not split across A/B
    controls[1,0,0] = 128
    controls[1,1,0] = 200
    counts = np.zeros((1,128),np.uint32)
    counts[0,:2] = [128,129 if late_overflow else 128]
    plan, fatal = module.pallas_collector_preflight(
        jnp.asarray(controls),jnp.asarray(counts),capacity=256,interpret=True)
    expected = np.zeros((2,4,128),np.uint32)
    if not late_overflow:
        expected[0,:,0] = [1,0,128,1]
        expected[1,:,0] = [0,128,128,1]
    # A rejected partition has no usable reservation, even for earlier shards.
    np.testing.assert_array_equal(plan,expected)
    flag = np.zeros((1,128),np.uint32)
    flag[0,0] = int(late_overflow)
    np.testing.assert_array_equal(fatal,flag)
