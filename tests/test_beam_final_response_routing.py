import numpy as np
import jax.numpy as jnp
import pytest


def inputs():
    requests = np.full((4,128), 0xffffffff, np.uint32)
    requests[3,:3] = [2 | (23<<16), 0, 1 | (7<<16)]
    control = np.zeros((2,128), np.uint32)
    control[0,0] = 3
    validation = np.zeros((2,128), np.uint32)
    validation[1,0] = 0xffffffff  # no-invalid sentinel is not an error
    return requests, control, validation


def test_response_routing_retains_return_rank_and_masks_poison_tail():
    from tpu_beam_search.beam_final_response_routing import pallas_final_response_routing
    ranks,valid,error = map(np.asarray,pallas_final_response_routing(
        *map(jnp.asarray,inputs()),world_size=3,interpret=True))
    np.testing.assert_array_equal(ranks[0,:3],[2,0,1])
    np.testing.assert_array_equal(valid[0,:3],[1,1,1])
    assert not ranks[:,3:].any() and not valid[:,3:].any()
    assert not error.any()


@pytest.mark.parametrize('failure',['receive','materialize','rank','reserved','count'])
def test_response_routing_masks_whole_batch_on_any_error(failure):
    from tpu_beam_search.beam_final_response_routing import pallas_final_response_routing
    requests,control,validation=inputs()
    if failure=='receive': control[1,0]=7
    elif failure=='materialize': validation[0,0]=0xffffffff
    elif failure=='rank': requests[3,2]=3
    elif failure=='reserved': requests[3,2]=1<<24
    elif failure=='count': control[0,0]=129
    ranks,valid,error=map(np.asarray,pallas_final_response_routing(
        *map(jnp.asarray,(requests,control,validation)),world_size=3,interpret=True))
    assert not ranks.any() and not valid.any()
    assert error[0,0]==1 and not error[0,1:].any()


def test_response_routing_empty_rank_ignores_invalid_payload():
    from tpu_beam_search.beam_final_response_routing import pallas_final_response_routing
    requests,control,validation=inputs()
    control[0,0]=0
    ranks,valid,error=map(np.asarray,pallas_final_response_routing(
        *map(jnp.asarray,(requests,control,validation)),world_size=3,interpret=True))
    assert not ranks.any() and not valid.any() and not error.any()
