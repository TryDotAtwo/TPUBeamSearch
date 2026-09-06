import numpy as np
import jax.numpy as jnp


def test_sorted_intervals_cross_tiles_and_empty_ranks():
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    ranks = np.full((1,256),0xffffffff,np.uint32)
    valid = np.zeros((1,256),np.uint32)
    ranks[0,:129] = 2
    ranks[0,129:134] = 5
    valid[0,:134] = 1
    got = np.asarray(pallas_final_rank_intervals(jnp.asarray(ranks),jnp.asarray(valid),world_size=8,interpret=True))
    np.testing.assert_array_equal(got[0,:8],[0,0,0,129,129,129,134,134])
    np.testing.assert_array_equal(got[1,:8],[0,0,129,0,0,5,0,0])
    assert not got[2].any()
    assert not got[:2,8:].any()


def test_empty_and_live_invalid_rank_are_distinguished():
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    ranks = jnp.full((1,128),0xffffffff,jnp.uint32)
    valid = jnp.zeros_like(ranks)
    empty = np.asarray(pallas_final_rank_intervals(ranks,valid,world_size=8,interpret=True))
    assert not empty.any()
    invalid = np.asarray(pallas_final_rank_intervals(ranks,valid.at[0,0].set(1),world_size=8,interpret=True))
    assert invalid[2,0] == 1
    assert not invalid[:2].any()


def test_intervals_index_actual_grouped_payload_without_losing_identity():
    from tpu_beam_search.beam_final_group import pallas_group_final_records
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    payload = np.zeros((4,256),np.uint32)
    payload[0] = np.arange(256,dtype=np.uint32)
    payload[1] = 0x12345678
    ranks = np.full((1,256),0xffffffff,np.uint32)
    valid = np.zeros((1,256),np.uint32)
    slots = [0,1,127,128,129,255]
    ranks[0,slots] = [7,2,7,0,2,7]
    valid[0,slots] = 1
    grouped = pallas_group_final_records(*map(jnp.asarray,(payload,ranks,valid)),interpret=True)
    intervals = np.asarray(pallas_final_rank_intervals(grouped[4:5],grouped[6:7],world_size=8,interpret=True))
    assert not intervals[2].any()
    output = np.asarray(grouped)
    for rank in range(8):
        start,count = map(int,intervals[:2,rank])
        expected = [i for i in slots if ranks[0,i] == rank]
        np.testing.assert_array_equal(output[0,start:start+count],expected)
        np.testing.assert_array_equal(output[1,start:start+count],np.full(count,0x12345678,np.uint32))
    assert int(intervals[1].sum()) == len(slots)
