import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('reject',[False,True])
def test_delivery_shares_target_indices_between_requests_and_history(reject):
    from tpu_beam_search.beam_final_delivery import pallas_final_delivery_plan
    packed = np.zeros((11,128),np.uint32)
    packed[4,:4] = [11,22,33,44]
    packed[5,:4] = [1,2,3,4]
    packed[7,:4] = [(6<<16)|(4<<8)|23,(2<<16)|(5<<8)|7,9,(3<<16)|1]
    packed[8,:4] = [0,3,0,18]
    packed[10,:4] = [1,1,0,1]
    keep = jnp.zeros((2,128),jnp.uint32).at[0,0].set(19)
    prior = jnp.zeros((1,128),jnp.uint32).at[0,0].set(int(reject))
    result = pallas_final_delivery_plan(jnp.asarray(packed),keep,prior,
                                      world_size=8,interpret=True)
    assert int(result.error[0,0]) == int(reject)
    if reject:
        assert not np.asarray(result.valid).any()
        assert not np.asarray(result.history).any()
        assert not np.asarray(result.target_counts).any()
    else:
        np.testing.assert_array_equal(np.flatnonzero(result.valid[0]),[0,1,3])
        np.testing.assert_array_equal(np.asarray(result.destinations)[0,[0,1,3]],[0,1,7])
        np.testing.assert_array_equal(np.asarray(result.sources)[0,[0,1,3]],[6,2,3])
        np.testing.assert_array_equal(np.asarray(result.requests)[2,[0,1,3]],[0,0,1])
        np.testing.assert_array_equal(np.asarray(result.requests)[3,[0,1,3]],
                                      [(23<<16),(7<<16)|1,(1<<16)|7])
        np.testing.assert_array_equal(np.asarray(result.history)[:2,[0,1,3]],packed[4:6,[0,1,3]])
        np.testing.assert_array_equal(np.asarray(result.history)[2,[0,1,3]],packed[7,[0,1,3]])
        np.testing.assert_array_equal(result.history[3],result.requests[2])
        np.testing.assert_array_equal(result.history[4],result.valid[0])
        np.testing.assert_array_equal(np.asarray(result.target_counts)[0,:8],[3,2,3,2,2,3,2,2])


def test_zero_keep_cannot_revive_a_marked_record_with_zero_index():
    from tpu_beam_search.beam_final_delivery import pallas_final_delivery_plan
    packed = jnp.zeros((11,128),jnp.uint32).at[10,0].set(1)
    result = pallas_final_delivery_plan(packed,jnp.zeros((2,128),jnp.uint32),
        jnp.zeros((1,128),jnp.uint32),world_size=8,interpret=True)
    assert not np.asarray(result.valid).any()
    assert not np.asarray(result.history).any()
    assert not np.asarray(result.target_counts).any()
