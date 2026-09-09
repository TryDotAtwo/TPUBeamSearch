import numpy as np
import pytest


def test_completed_history_soa_preserves_parent64_route_and_ignores_padding():
    from tpu_beam_search import beam_history
    assert hasattr(beam_history,'decode_history_soa')
    data = np.full((5,128),0xffffffff,np.uint32)
    data[4] = 0
    data[:,0] = [9,1,(6<<16)|(4<<8)|23,2,1]
    records = list(beam_history.decode_history_soa(data,world_size=8,move_count=24))
    assert records == [(2,beam_history.HistoryEntry((1<<32)+9,(6<<16)|(4<<8)|23))]


@pytest.mark.parametrize('plane,value',[(4,2),(2,8<<16),(2,24)])
def test_malformed_live_history_cannot_publish_partial_depth(plane,value):
    from tpu_beam_search import beam_history
    assert hasattr(beam_history,'decode_history_soa')
    data = np.zeros((5,128),np.uint32)
    data[:,0] = [0,0,1,0,1]
    data[:,1] = [0,0,1,1,1]
    data[plane,1] = value
    store = beam_history.RankHistoryStore(world_size=8)
    with pytest.raises(ValueError):
        store.append_all_rank_layer(
            [beam_history.decode_history_soa(data,world_size=8,move_count=24)]+[[]]*7,
            target_counts=[2]+[0]*7,depth=0)
    with pytest.raises(IndexError):
        store.read_entry(0,0,0)


@pytest.mark.parametrize('data', [np.zeros((4,128),np.uint32),
                                np.zeros((5,128),np.int32), [[0]*5]])
def test_decode_rejects_non_host_abi(data):
    from tpu_beam_search.beam_history import decode_history_soa
    with pytest.raises(ValueError):
        list(decode_history_soa(data,world_size=8,move_count=24))


def test_pallas_projection_to_host_publication_preserves_original_source():
    import jax.numpy as jnp
    from tpu_beam_search.beam_final_history import pallas_final_history_records
    from tpu_beam_search.beam_history import decode_history_soa,RankHistoryStore,HistoryEntry
    meta = np.full((8,128),0xffffffff,np.uint32)
    meta[4:6,0] = [9,1]
    meta[7,0] = (6<<16)|(4<<8)|23
    valid = np.zeros((1,128),np.uint32)
    valid[0,0] = 1
    records = np.asarray(pallas_final_history_records(jnp.asarray(meta),
        jnp.zeros((1,128),jnp.uint32),jnp.asarray(valid),interpret=True))
    store = RankHistoryStore(world_size=8)
    # Balanced destination2 is deliberately different from original source6.
    ranks = [[] for _ in range(8)]
    ranks[2] = decode_history_soa(records,world_size=8,move_count=24)
    store.append_all_rank_layer(ranks,target_counts=[0,0,1,0,0,0,0,0],depth=0)
    assert store.read_entry(2,0,0) == HistoryEntry((1<<32)+9,(6<<16)|(4<<8)|23)
