"""Logical-rank host routing with Pallas interpretation, NOT remote TPU replay."""
import numpy as np
import jax.numpy as jnp


def test_final_plan_materialization_and_history_agree_after_relocation():
    from tpu_beam_search.beam_final_plan import pallas_final_history_plan
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    from tpu_beam_search.beam_history import HistoryEntry,RankHistoryStore
    parents=np.zeros((2,2,128),np.uint8)
    parents[0,0,:3],parents[0,1,:3]=[0,1,2],[2,1,0]
    parents[1,0,:3],parents[1,1,:3]=[1,2,0],[2,0,1]
    generators=np.tile(np.arange(128,dtype=np.int32),(2,1))
    generators[1,:3]=[1,2,0]
    meta=np.zeros((8,128),np.uint32)
    meta[4,:3]=[0,1,1]
    meta[7,:3]=[(1<<16),1,(1<<16)|1]
    indices=np.zeros((2,128),np.uint32)
    indices[0,:]=3
    indices[0,:3]=[0,1,2]
    boundaries=np.zeros((2,128),np.uint32)
    boundaries[0,:3]=[0,2,3]
    plan=pallas_final_history_plan(*map(jnp.asarray,(meta,indices,boundaries)),world_size=2,interpret=True)
    requests,sources,valid,history,destinations=map(np.asarray,plan)
    received=[[],[]]
    # Explicit host routing is a fixture adapter, not the production data plane.
    for source in range(2):
        slots=[i for i in range(128) if valid[0,i] and sources[0,i]==source]
        packed=np.zeros((4,128),np.uint32)
        packed[:,:len(slots)]=requests[:,slots]
        wire,errors=pallas_materialize_final(jnp.asarray(parents[source]),
            jnp.asarray(generators),jnp.asarray(packed),jnp.array([len(slots)],jnp.uint32),
            jnp.array([2],jnp.uint32),state_len=120,interpret=True)
        assert int(errors[0,0])==0
        for row,slot in enumerate(slots):
            received[int(requests[3,slot]&65535)].append(np.asarray(wire[row]))
    store=RankHistoryStore(world_size=2)
    for rank,count in enumerate((2,1)):
        wire=np.zeros((128,128),np.uint8)
        wire[:count]=received[rank]
        frontier,errors=pallas_scatter_final_responses(jnp.zeros((count,128),jnp.uint8),
            jnp.asarray(wire),jnp.array([count],jnp.uint32),state_len=120,interpret=True)
        assert int(errors[0,0])==0
        slots=[i for i in range(128) if valid[0,i] and destinations[0,i]==rank]
        store.append_rank_layer(rank,[(int(history[3,i]),HistoryEntry(int(history[0,i]),int(history[2,i]))) for i in slots],target_count=count)
        for slot in slots:
            local=int(history[3,slot])
            entry=store.read_entry(rank,0,local)
            source=entry.route_packed>>16
            move=entry.route_packed&255
            expected=parents[source,entry.parent_idx,generators[move]].copy()
            expected[120:]=0
            np.testing.assert_array_equal(frontier[local],expected)
