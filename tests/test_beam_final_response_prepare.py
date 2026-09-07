import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('failed',[False,True])
def test_response_prepare_groups_exact_wire_and_zeroes_send_counts_on_failure(failed):
    from tpu_beam_search.beam_final_response_routing import pallas_prepare_final_response_exchange
    wire=np.arange(128*128,dtype=np.uint8).reshape(128,128)
    requests=np.zeros((4,128),np.uint32)
    requests[3,:5]=[2,0,2,1,0]
    control=np.zeros((2,128),np.uint32)
    control[0,0]=5
    validation=np.zeros((2,128),np.uint32)
    validation[0,0]=int(failed)
    grouped,intervals,error=map(np.asarray,pallas_prepare_final_response_exchange(
        *map(jnp.asarray,(wire,requests,control,validation)),world_size=3,interpret=True))
    if failed:
        assert not grouped[34].any()
        assert not intervals.any()
        assert error[0,0]==1
    else:
        # Literal stable order by return rank, then original slot.
        order=[1,4,3,0,2]
        words=wire.copy().view('<u4').T
        np.testing.assert_array_equal(grouped[:32,:5],words[:,order])
        np.testing.assert_array_equal(grouped[32,:5],[0,0,1,2,2])
        np.testing.assert_array_equal(grouped[33,:5],order)
        np.testing.assert_array_equal(grouped[34],np.arange(128)<5)
        np.testing.assert_array_equal(intervals[:2,:3],[[0,2,3],[2,1,2]])
        assert not intervals[2].any() and not error.any()
