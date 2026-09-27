from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def test_eight_rank_final_depth_traces_selection_and_both_streams():
    from tpu_beam_search.beam_final_depth import make_final_depth_call
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState
    u32=lambda shape:jax.ShapeDtypeStruct(shape,jnp.uint32)
    state=FinalMaterializationState(jax.ShapeDtypeStruct((128,160),jnp.uint8),
        u32((1,5,128)),u32((1,128)),u32((2,128)),u32((1,128)),
        u32((2,128)),u32((1,128)))
    call=make_final_depth_call(SimpleNamespace(size=8),state_len=150,move_count=30)
    trace=jax.make_jaxpr(call,axis_env=[('core',8)])(
        u32((1,8,128)),u32((1,8,128)),u32((1,8,128)),
        u32((1,)),u32((2,128)),u32((1,128)),
        jax.ShapeDtypeStruct((128,256),jnp.uint8),
        jax.ShapeDtypeStruct((30,256),jnp.int32),state)
    assert [value.shape for value in trace.out_avals]==[
        (128,160),(1,5,128),(1,128),(2,128),(1,128),(2,128),(1,128),
        (8,),(2,128),(2,128),(1,128),(1,128)]


def test_one_rank_frozen_final_depth_selects_and_materializes_in_physical_order():
    from tpu_beam_search.beam_final_depth import make_final_depth_call
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState
    from tpu_beam_search.beam_final_publication import (
        PublishedBeamDepth,commit_final_epoch_states)
    from tpu_beam_search.beam_history import HistoryEntry,RankHistoryStore
    a=np.zeros((1,8,128),np.uint32)
    b=np.zeros_like(a)
    a[:,6],b[:,6]=0xffffffff,0xffffffff
    a[0,6,0]=b[0,6,0]=5
    a[0,4,0],b[0,4,0]=1,0
    controls=np.zeros((1,8,128),np.uint32)
    controls[0,:2,0]=1
    parents=np.zeros((2,256),np.uint8)
    parents[:,:3]=[[1,2,3],[4,5,6]]
    generators=np.arange(256,dtype=np.int32)[None,:]
    zero=jnp.zeros((1,128),jnp.uint32)
    state=FinalMaterializationState(jnp.zeros((128,160),jnp.uint8),
        jnp.zeros((1,5,128),jnp.uint32),zero,jnp.zeros((2,128),jnp.uint32),
        zero,jnp.zeros((2,128),jnp.uint32),zero)
    beam=jnp.zeros((2,128),jnp.uint32).at[0,0].set(2)
    result,target_counts,keep,_,response_error,history_error=(
        make_final_depth_call(SimpleNamespace(size=1),state_len=150,
                              move_count=1,interpret=True)(
            jnp.asarray(a),jnp.asarray(b),jnp.asarray(controls),
            jnp.array([5],jnp.uint32),beam,zero,jnp.asarray(parents),
            jnp.asarray(generators),state))
    assert int(keep[0,0])==2 and int(target_counts[0])==2
    assert not np.asarray(result.error).any()
    assert not np.asarray(response_error).any()
    assert not np.asarray(history_error).any()
    np.testing.assert_array_equal(np.asarray(result.frontier)[:2,:3],
                                  [[4,5,6],[1,2,3]])
    current=PublishedBeamDepth(0,(jnp.zeros((128,160),jnp.uint8),),(2,),
                               RankHistoryStore(world_size=1))
    published=commit_final_epoch_states(current,states_by_rank=(result,),
        target_counts=target_counts,move_count=1,interpret=True)
    assert published.depth==1 and published.counts==(2,)
    assert published.history.read_entry(0,0,0)==HistoryEntry(1,0)
    assert published.history.read_entry(0,0,1)==HistoryEntry(0,0)
