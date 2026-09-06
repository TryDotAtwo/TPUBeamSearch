import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('chunk',[0,1,2,0xffffffff])
def test_chunk_preserves_unaligned_ranges_and_zero_tail(chunk):
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    payload=np.arange(4*384,dtype=np.uint32).reshape(4,384)
    intervals=np.zeros((3,128),np.uint32)
    intervals[0,:4]=[0,1,130,384]
    intervals[1,:4]=[1,129,254,0]
    wire,control=map(np.asarray,pallas_pack_final_chunk(jnp.asarray(payload),jnp.asarray(intervals),
        jnp.array([chunk],jnp.uint32),world_size=4,interpret=True))
    for rank in range(4):
        start,count=map(int,intervals[:2,rank])
        offset=min(chunk*128,count)
        length=min(128,count-offset)
        expected=np.zeros((4,128),np.uint32)
        expected[:,:length]=payload[:,start+offset:start+offset+length]
        np.testing.assert_array_equal(wire[rank],expected)
        assert control[rank,0,0]==length
        assert not control[rank,1].any()


@pytest.mark.parametrize('bad', ['bounds','rank'])
def test_bad_interval_blocks_all_payload_reads(bad):
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    intervals=np.zeros((3,128),np.uint32)
    intervals[1,0]=1
    if bad=='bounds':
        intervals[0,1],intervals[1,1]=0xffffffff,2
    else:
        intervals[2,0]=1
    wire,control=map(np.asarray,pallas_pack_final_chunk(jnp.ones((4,128),jnp.uint32),jnp.asarray(intervals),
        jnp.array([0],jnp.uint32),world_size=2,interpret=True))
    assert not wire.any()
    assert not control[:,0].any()
    np.testing.assert_array_equal(control[:,1,0],[1,1])


def test_group_intervals_and_chunks_preserve_every_valid_record():
    from tpu_beam_search.beam_final_group import pallas_group_final_records
    from tpu_beam_search.beam_final_intervals import pallas_final_rank_intervals
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    payload=np.arange(4*256,dtype=np.uint32).reshape(4,256)
    ranks=np.full((1,256),3,np.uint32)
    ranks[0,::17]=1
    valid=np.ones((1,256),np.uint32)
    valid[0,::31]=0
    grouped=pallas_group_final_records(*map(jnp.asarray,(payload,ranks,valid)),interpret=True)
    intervals=pallas_final_rank_intervals(grouped[4:5],grouped[6:7],world_size=4,interpret=True)
    received=[[] for _ in range(4)]
    for index in range(3):
        wire,control=map(np.asarray,pallas_pack_final_chunk(grouped[:4],intervals,
            jnp.array([index],jnp.uint32),world_size=4,interpret=True))
        assert not control[:,1].any()
        for rank in range(4):
            received[rank].append(wire[rank,:,:int(control[rank,0,0])])
    for rank in range(4):
        expected=payload[:,(valid[0]!=0)&(ranks[0]==rank)]
        np.testing.assert_array_equal(np.concatenate(received[rank],axis=1),expected)


def test_cross_tile_dma_with_interpreter_race_detection():
    from jax.experimental.pallas import tpu as pltpu
    from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
    payload=np.arange(4*256,dtype=np.uint32).reshape(4,256)
    intervals=np.zeros((3,128),np.uint32)
    intervals[0,0],intervals[1,0]=127,2
    wire,control=map(np.asarray,pallas_pack_final_chunk(jnp.asarray(payload),jnp.asarray(intervals),
        jnp.array([0],jnp.uint32),world_size=1,
        interpret=pltpu.InterpretParams(detect_races=True)))
    np.testing.assert_array_equal(wire[0,:,:2],payload[:,127:129])
    assert not wire[0,:,2:].any()
    assert control[0,0,0]==2
