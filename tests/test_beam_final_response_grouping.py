"""Local response routing composition; not physical inter-device execution."""
import jax.numpy as jnp
import numpy as np


def test_grouped_response_transport_preserves_target_and_valid_prefix():
    from tpu_beam_search.beam_final_transport import pallas_wire_to_planes, pallas_planes_to_wire
    from tpu_beam_search.beam_final_response import pallas_pack_response, pallas_unpack_response
    from tpu_beam_search.beam_final_group import pallas_group_final_records
    n, live = 128, 17
    states = np.random.default_rng(613).integers(0,256,(n,128),dtype=np.uint8)
    targets = (np.arange(n,dtype=np.uint32)+0xfeed0000)[None,:]
    ranks = ((np.arange(n,dtype=np.uint32)*3)%8)[None,:]
    valid = (np.arange(n)<live).astype(np.uint32)[None,:]
    wire = pallas_pack_response(jnp.asarray(states),jnp.asarray(targets),state_len=120,interpret=True)
    planes = pallas_wire_to_planes(wire,interpret=True)
    grouped = pallas_group_final_records(planes,jnp.asarray(ranks),jnp.asarray(valid),interpret=True)
    restored = pallas_planes_to_wire(grouped[:32],interpret=True)
    actual, decoded_targets = pallas_unpack_response(restored,state_len=120,interpret=True)
    order = np.argsort(ranks[0,:live],kind='stable')
    np.testing.assert_array_equal(actual[:live,:120],states[order,:120])
    np.testing.assert_array_equal(decoded_targets[0,:live],targets[0,order])
    np.testing.assert_array_equal(grouped[32,:live],ranks[0,order])
    np.testing.assert_array_equal(grouped[34],valid[0])
    assert not np.asarray(actual[:,120:]).any()

    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    # Include empty destinations 8 and 9 without changing live return ranks.
    intervals = pallas_final_rank_intervals(grouped[32:33],grouped[34:35],world_size=10,interpret=True)
    chunks, controls = pallas_pack_final_chunk(grouped[:32],intervals,jnp.array([0],jnp.uint32),
        world_size=10,interpret=True)
    assert not np.asarray(controls[:,1]).any()
    for rank in range(10):
        original = np.flatnonzero(ranks[0,:live] == rank)
        count = int(controls[rank,0,0])
        assert count == len(original)
        np.testing.assert_array_equal(chunks[rank,:,:count],np.asarray(planes)[:,original])
        assert not np.asarray(chunks[rank,:,count:]).any()
    empty, controls = pallas_pack_final_chunk(grouped[:32],intervals,jnp.array([1],jnp.uint32),
        world_size=10,interpret=True)
    assert not np.asarray(empty).any()
    assert not np.asarray(controls).any()
