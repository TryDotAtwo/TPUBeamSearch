import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('bound',[1,2])
def test_destination_validation_reaches_every_response_chunk_control(bound):
    from tpu_beam_search.beam_final_receive import pallas_materialize_final_snapshots
    from tpu_beam_search.beam_final_response_routing import pallas_prepare_final_response_exchange
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    requests = np.full((1,4,128),0xffffffff,np.uint32)
    requests[0,:,0] = [0,0,1,1]
    counts = np.zeros((1,1,128),np.uint32)
    counts[0,0,0] = 1
    wire,validation,control,packed = pallas_materialize_final_snapshots(
        jnp.full((1,128),11,jnp.uint8),jnp.arange(128,dtype=jnp.int32)[None,:],
        jnp.asarray(requests),jnp.asarray(counts),jnp.zeros((1,128),jnp.uint32),
        jnp.array([99],jnp.uint32),state_len=120,interpret=True,
        return_counts=jnp.array([2,bound],jnp.uint32))
    grouped,intervals,error = pallas_prepare_final_response_exchange(
        wire,packed,control,validation,world_size=2,interpret=True)
    expected_wire = np.zeros((128,128),np.uint8)
    expected_wire[0,:120] = 11
    expected_wire[0,120] = 1
    for epoch in (0,1):
        payload,send = map(np.asarray,pallas_pack_final_chunk(
            grouped[:32],intervals,jnp.array([epoch],jnp.uint32),
            world_size=2,prior_error=error,interpret=True))
        expected_payload = np.zeros((2,32,128),np.uint32)
        expected_control = np.zeros((2,2,128),np.uint32)
        if bound == 1:
            expected_control[:,1,0] = 1
        elif epoch == 0:
            expected_control[1,0,0] = 1
            expected_payload[1] = expected_wire.view('<u4').T
        np.testing.assert_array_equal(payload,expected_payload)
        np.testing.assert_array_equal(send,expected_control)


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
    # Exercise the actual transport control consumer, not just zero intervals.
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    packets,controls=map(np.asarray,pallas_pack_final_chunk(
        jnp.asarray(grouped[:32]),jnp.asarray(intervals),jnp.array([0],jnp.uint32),
        world_size=3,prior_error=jnp.asarray(error),interpret=True))
    if failed:
        assert not grouped[34].any()
        assert not intervals.any()
        assert error[0,0]==1
        assert not packets.any() and not controls[:,0].any()
        np.testing.assert_array_equal(controls[:,1,0],[1,1,1])
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
        np.testing.assert_array_equal(controls[:,0,0],[2,1,2])
        assert not controls[:,1].any()
        for rank,indices in enumerate(([1,4],[3],[0,2])):
            np.testing.assert_array_equal(packets[rank,:,:len(indices)],words[:,indices])
            assert not packets[rank,:,len(indices):].any()
