import numpy as np
import jax.numpy as jnp


def test_request_and_history_share_balanced_targets_and_validity():
    from tpu_beam_search.beam_final_plan import pallas_final_history_plan
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore
    meta=np.zeros((8,128),np.uint32)
    meta[4,:3]=[9,7,5]
    meta[5,:3]=1
    meta[7,:3]=(1<<16)|(4<<8)|23
    indices=np.zeros((2,128),np.uint32)
    indices[0,:]=3
    indices[0,:3]=np.arange(3)
    boundaries=np.zeros((2,128),np.uint32)
    boundaries[0,:3]=[0,2,3]
    req,source,valid,history,ranks=pallas_final_history_plan(
        jnp.asarray(meta),jnp.asarray(indices),jnp.asarray(boundaries),
        world_size=2,interpret=True)
    req,source,valid,history,ranks=map(np.asarray,(req,source,valid,history,ranks))
    np.testing.assert_array_equal(history[3],req[2])
    np.testing.assert_array_equal(history[4],valid[0])
    np.testing.assert_array_equal(ranks[0,:3],[0,0,1])
    store=RankHistoryStore(world_size=2)
    for rank,count in enumerate((2,1)):
        rows=[(int(history[3,i]),HistoryEntry(int(history[0,i])|(int(history[1,i])<<32),int(history[2,i])))
              for i in range(128) if valid[0,i] and ranks[0,i]==rank]
        store.append_rank_layer(rank,rows,target_count=count)
    assert store.read_entry(1,0,0).parent_idx == (1<<32)|5
    assert store.read_entry(1,0,0).route_packed == int(meta[7,2])
