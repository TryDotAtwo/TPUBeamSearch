"""Final epoch outputs enter the existing atomic publication boundary."""
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('late_error',[False,True])
def test_private_epoch_outputs_publish_frontier_and_history_together(late_error):
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState
    from tpu_beam_search.beam_final_publication import (
        PublishedBeamDepth, commit_final_epoch_states)
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore

    old=jnp.zeros((128,160),jnp.uint8)
    current=PublishedBeamDepth(0,(old,),(1,),RankHistoryStore(world_size=1))
    private=jnp.zeros_like(old).at[0,:3].set(jnp.array([1,2,3],jnp.uint8))
    tiled=np.zeros((1,5,128),np.uint32)
    tiled[0,:,0]=[0,0,1,0,1]
    zero=jnp.zeros((1,128),jnp.uint32)
    state=FinalMaterializationState(private,jnp.asarray(tiled),zero,
        jnp.zeros((2,128),jnp.uint32),zero,jnp.zeros((2,128),jnp.uint32),
        zero.at[0,0].set(1) if late_error else zero)
    published=commit_final_epoch_states(current,states_by_rank=(state,),
        target_counts=jnp.array([1],jnp.uint32),move_count=30,interpret=True)
    assert current.depth==0
    np.testing.assert_array_equal(current.frontiers[0],old)
    if late_error:
        assert published is current
    else:
        assert published.depth==1 and published.counts==(1,)
        np.testing.assert_array_equal(published.frontiers[0],private)
        assert published.history.read_entry(0,0,0)==HistoryEntry(0,1)
