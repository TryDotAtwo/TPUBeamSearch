import jax.numpy as jnp
import jax
import numpy as np
import pytest


@pytest.mark.parametrize('failure', [None, 'late_history', 'target', 'receive_count'])
def test_received_requests_authorize_responses_only_after_all_local_gates(failure):
    from tpu_beam_search.beam_final_response_plan import pallas_final_response_plan
    from tpu_beam_search.beam_final_transport import pallas_planes_to_wire

    parents = np.zeros((2,128), np.uint8)
    parents[0,:3] = [3,4,5]
    parents[1,:3] = [6,7,8]
    generators = np.tile(np.arange(128,dtype=np.int32), (2,1))
    generators[1,:3] = [2,0,1]
    snapshots = np.full((2,4,128), 0xffffffff, np.uint32)
    snapshots[0,:,0] = [1,0,1,1 | (1<<16)]
    snapshots[1,:,0] = [0,0,0,0]
    counts = np.zeros((2,1,128), np.uint32)
    counts[:,0,0] = 1
    error = np.zeros((1,128), np.uint32)
    targets = np.array([1,2], np.uint32)
    if failure == 'late_history': error[0,0] = 1
    if failure == 'target': targets[1] = 1
    if failure == 'receive_count': counts[1,0,0] = 129
    grouped,intervals,error = map(np.asarray, pallas_final_response_plan(
        *map(jnp.asarray,(parents,generators,snapshots,counts,error,targets)),
        state_len=120,world_size=2,interpret=True))
    assert error[0,0] == int(failure is not None)
    assert not error[0,1:].any()
    if failure is not None:
        assert not intervals[1].any()
        assert not grouped[-1].any()
    else:
        np.testing.assert_array_equal(intervals[1,:2], [1,1])
        np.testing.assert_array_equal(grouped[-3,:2], [0,1])
        wire = np.asarray(pallas_planes_to_wire(jnp.asarray(grouped[:32]),interpret=True))
        expected = np.zeros((256,128), np.uint8)
        expected[0,:3] = [3,4,5]
        expected[1,:3] = [8,6,7]
        expected[1,120] = 1
        np.testing.assert_array_equal(wire,expected)


def test_response_plan_rejects_incomplete_destination_capacity_vector():
    from tpu_beam_search.beam_final_response_plan import pallas_final_response_plan
    with pytest.raises(ValueError,match='geometry'):
        pallas_final_response_plan(jnp.zeros((1,128),jnp.uint8),
            jnp.arange(128,dtype=jnp.int32)[None],
            jnp.zeros((2,4,128),jnp.uint32),jnp.zeros((2,1,128),jnp.uint32),
            jnp.zeros((1,128),jnp.uint32),jnp.ones((1,),jnp.uint32),
            state_len=120,world_size=2,interpret=True)


def test_response_plan_eight_rank_shapes_trace_without_count_readback():
    from tpu_beam_search.beam_final_response_plan import pallas_final_response_plan
    def call(*args):
        return pallas_final_response_plan(*args,state_len=150,world_size=8)
    specs = [((16,256),jnp.uint8),((30,256),jnp.int32),
             ((8,4,128),jnp.uint32),((8,1,128),jnp.uint32),
             ((1,128),jnp.uint32),((8,),jnp.uint32)]
    traced = jax.make_jaxpr(call)(*(jax.ShapeDtypeStruct(*s) for s in specs))
    assert [(a.shape,a.dtype) for a in traced.out_avals] == [
        ((67,1024),jnp.dtype('uint32')),((3,128),jnp.dtype('uint32')),
        ((1,128),jnp.dtype('uint32'))]
