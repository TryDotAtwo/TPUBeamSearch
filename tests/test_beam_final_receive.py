import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('bad_count',[129,0x80000000,0xffffffff])
def test_received_count_overflow_cannot_wrap_into_valid_total(bad_count):
    from tpu_beam_search.beam_final_receive import pallas_compact_final_received
    counts = jnp.zeros((2,1,128),jnp.uint32).at[:,0,0].set(
        jnp.array([128,bad_count],jnp.uint32))
    packed,control = pallas_compact_final_received(jnp.ones((2,4,128),jnp.uint32),
        counts,jnp.zeros((1,128),jnp.uint32),interpret=True)
    np.testing.assert_array_equal(packed,np.zeros((4,256),np.uint32))
    expected = np.zeros((2,128),np.uint32)
    expected[1,0] = 1
    np.testing.assert_array_equal(control,expected)


@pytest.mark.parametrize('destination_count,accepted',[(2,True),(1,False),(0,False)])
def test_received_mixed_rank_bounds_survive_compaction(destination_count,accepted):
    from tpu_beam_search.beam_final_receive import pallas_materialize_final_snapshots
    requests = np.full((2,4,128),0xffffffff,np.uint32)
    requests[0,:,0] = [0,0,1,0]
    requests[1,:,0] = [0,0,1,1]
    counts = np.zeros((2,1,128),np.uint32)
    counts[:,0,0] = 1
    wire,errors,control,packed = map(np.asarray,pallas_materialize_final_snapshots(
        jnp.full((1,128),9,jnp.uint8),jnp.arange(128,dtype=jnp.int32)[None,:],
        jnp.asarray(requests),jnp.asarray(counts),jnp.zeros((1,128),jnp.uint32),
        jnp.array([99],jnp.uint32),state_len=120,interpret=True,
        return_counts=jnp.array([2,destination_count],jnp.uint32)))
    expected = np.zeros((256,128),np.uint8)
    if accepted:
        expected[:2,:120] = 9
        expected[:2,120] = 1
    np.testing.assert_array_equal(wire,expected)
    assert errors[0,0] == (0 if accepted else 1)
    assert errors[1,0] == (0xffffffff if accepted else 1)
    assert control[0,0] == 2 and control[1,0] == 0
    np.testing.assert_array_equal(packed[:,:2],[[0,0],[0,0],[1,1],[0,1]])
    assert not packed[:,2:].any()


def test_snapshot_holes_are_removed_before_materialization():
    from tpu_beam_search.beam_final_receive import pallas_compact_final_received
    snapshots=np.arange(3*4*128,dtype=np.uint32).reshape(3,4,128)+1
    counts=np.zeros((3,1,128),np.uint32)
    counts[:,0,0]=[2,0,3]
    packed,control=map(np.asarray,pallas_compact_final_received(jnp.asarray(snapshots),jnp.asarray(counts),jnp.zeros((1,128),jnp.uint32),interpret=True))
    expected=np.concatenate((snapshots[0,:,:2],snapshots[2,:,:3]),axis=1)
    np.testing.assert_array_equal(packed[:,:5],expected)
    assert packed.shape==(4,512)
    assert not packed[:,5:].any()
    assert control[0,0]==5 and not control[1].any()


def test_common_error_masks_all_snapshots():
    from tpu_beam_search.beam_final_receive import pallas_compact_final_received
    payload=jnp.ones((1,4,128),jnp.uint32)
    counts=jnp.zeros((1,1,128),jnp.uint32).at[0,0,0].set(128)
    packed,control=map(np.asarray,pallas_compact_final_received(payload,counts,jnp.ones((1,128),jnp.uint32),interpret=True))
    assert not packed.any()
    assert control[0,0]==0 and control[1,0]==1


def test_received_requests_materialize_without_host_filtering():
    from tpu_beam_search.beam_final_receive import pallas_materialize_final_snapshots
    parents=np.zeros((3,128),np.uint8)
    parents[:,:3]=[[0,1,2],[2,0,1],[1,2,0]]
    generators=np.tile(np.arange(128,dtype=np.int32),(2,1))
    generators[1,:3]=[1,2,0]
    requests=np.zeros((2,4,128),np.uint32)
    requests[0,:,0]=[2,0,1,1<<16]
    requests[1,:,0]=[0,0,0,0]
    counts=np.zeros((2,1,128),np.uint32)
    counts[:,0,0]=1
    wire,errors,control,packed=map(np.asarray,pallas_materialize_final_snapshots(
        *map(jnp.asarray,(parents,generators,requests,counts)),
        jnp.zeros((1,128),jnp.uint32),jnp.array([2],jnp.uint32),state_len=120,interpret=True))
    assert errors[0,0]==0 and not control[1].any()
    np.testing.assert_array_equal(packed[:,:2],np.stack((requests[0,:,0],requests[1,:,0]),axis=1))
    assert control[0,0]==2
    np.testing.assert_array_equal(wire[0,:120],parents[2,generators[1,:120]])
    np.testing.assert_array_equal(wire[1,:120],parents[0,:120])
    np.testing.assert_array_equal(wire[:2,120:124],[[1,0,0,0],[0,0,0,0]])
    assert not wire[2:].any()
