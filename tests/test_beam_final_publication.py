"""Single-host publication boundary; physical transport is gated separately."""
import jax.numpy as jnp
import numpy as np
import pytest


def test_staged_history_does_not_publish_into_old_snapshot():
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore
    old = RankHistoryStore(world_size=2)
    staged = old.stage_all_rank_layer(
        [[(0, HistoryEntry(3, 1))], []], target_counts=(1, 0), depth=0,
    )
    assert staged.read_entry(0, 0, 0) == HistoryEntry(3, 1)
    with pytest.raises(IndexError):
        old.read_entry(0, 0, 0)


def test_staged_history_retry_does_not_inherit_failed_depth():
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore
    root = RankHistoryStore(world_size=2)
    first = root.stage_all_rank_layer(
        [[(0, HistoryEntry(0, 1))], [(0, HistoryEntry(0, 2))]],
        target_counts=(1, 1), depth=0,
    )
    with pytest.raises(ValueError, match='missing history target'):
        first.stage_all_rank_layer(
            [[(0, HistoryEntry(0, 3))], []], target_counts=(1, 1), depth=1,
        )
    second = first.stage_all_rank_layer(
        [[(0, HistoryEntry(0, 4))], [(0, HistoryEntry(0, 5))]],
        target_counts=(1, 1), depth=1,
    )
    assert second.read_entry(0, 0, 0) == HistoryEntry(0, 1)
    assert second.read_entry(0, 1, 0) == HistoryEntry(0, 4)
    assert second.read_entry(1, 1, 0) == HistoryEntry(0, 5)
    with pytest.raises(IndexError):
        first.read_entry(0, 1, 0)


def _fixture():
    from tpu_beam_search.beam_final_publication import PublishedBeamDepth
    from tpu_beam_search.beam_history import RankHistoryStore
    old = tuple(jnp.zeros((128, 160), jnp.uint8) for _ in range(3))
    candidate = tuple(jnp.zeros((128, 160), jnp.uint8).at[:, :150].set(rank + 1)
                      for rank in range(3))
    records = [np.zeros((5, 128), np.uint32) for _ in range(3)]
    records[0][:, 0] = [3, 0, (2 << 16) | 4, 0, 1]
    records[0][:, 1] = [5, 0, 7, 1, 1]
    records[2][:, 0] = [6, 0, (1 << 16) | 2, 0, 1]
    current = PublishedBeamDepth(0, old, (8, 8, 8), RankHistoryStore(world_size=3))
    return current, candidate, records, jnp.array([2, 0, 1], jnp.uint32)


def test_completed_frontier_and_history_are_returned_as_one_new_snapshot():
    from tpu_beam_search.beam_final_publication import commit_final_publication
    from tpu_beam_search.beam_history import HistoryEntry
    current, candidate, records, counts = _fixture()
    result = commit_final_publication(
        current, frontier_by_rank=candidate,
        history_by_rank=tuple(map(jnp.asarray, records)), target_counts=counts,
        common_error=jnp.zeros((3, 1, 128), jnp.uint32),
        completed_work=candidate, move_count=30,
    )
    assert result.depth == 1 and result.counts == (2, 0, 1)
    assert all(a is b for a, b in zip(result.frontiers, candidate))
    assert result.history.read_entry(0, 0, 1) == HistoryEntry(5, 7)
    assert result.history.read_entry(2, 0, 0) == HistoryEntry(6, (1 << 16) | 2)
    with pytest.raises(IndexError):
        current.history.read_entry(0, 0, 0)
    assert current.depth == 0 and current.counts == (8, 8, 8)


@pytest.mark.parametrize('fault', ['empty_rank_error', 'duplicate', 'missing', 'source', 'parent'])
def test_late_failure_preserves_all_old_frontiers_and_history(fault):
    from tpu_beam_search.beam_final_publication import commit_final_publication
    current, candidate, records, counts = _fixture()
    errors = np.zeros((3, 1, 128), np.uint32)
    if fault == 'empty_rank_error':
        errors[1, 0, 0] = 1
    elif fault == 'duplicate':
        records[2][:, 1] = records[2][:, 0]
    elif fault == 'missing':
        records[2][4, 0] = 0
    elif fault == 'source':
        records[2][2, 0] = 3 << 16
    else:
        records[2][0, 0] = 8  # source rank 1 has only eight logical parents
    kwargs = dict(frontier_by_rank=candidate, history_by_rank=tuple(map(jnp.asarray, records)),
                  target_counts=counts, common_error=jnp.asarray(errors),
                  completed_work=candidate, move_count=30)
    if fault == 'empty_rank_error':
        assert commit_final_publication(current, **kwargs) is current
    else:
        with pytest.raises(ValueError):
            commit_final_publication(current, **kwargs)
    for rank in range(3):
        np.testing.assert_array_equal(current.frontiers[rank], 0)
        with pytest.raises(IndexError):
            current.history.read_entry(rank, 0, 0)


@pytest.mark.parametrize('state_len,storage_width,moves', [(120,128,24),(150,160,30)])
@pytest.mark.parametrize('late_error', [False, True])
def test_real_pallas_final_outputs_feed_one_publication(state_len,storage_width,moves,late_error):
    from types import SimpleNamespace
    from tpu_beam_search.beam_final_publication import PublishedBeamDepth, commit_final_publication
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    from tpu_beam_search.beam_final_response import pallas_unpack_response
    from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement
    from tpu_beam_search.beam_final_scatter import pallas_scatter_compact_final_responses
    from tpu_beam_search.beam_final_history import pallas_final_history_records
    from tpu_beam_search.beam_state_width_bridge import pallas_expand_state_rows, pallas_expand_generator_rows
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore

    parents = np.zeros((128,storage_width),np.uint8)
    parents[0,:state_len] = np.arange(state_len,dtype=np.uint8)
    parents[1,:state_len] = np.arange(state_len-1,-1,-1,dtype=np.uint8)
    generators = np.zeros((moves,storage_width),np.int32)
    generators[:,:state_len] = np.arange(state_len-1,-1,-1,dtype=np.int32)
    requests = np.zeros((4,128),np.uint32)
    requests[0,:2], requests[2,:2] = [1,0], [1,0]
    meta = np.zeros((8,128),np.uint32)
    meta[4,:2] = [1,0]
    valid = jnp.asarray((np.arange(128)<2).astype(np.uint32)[None,:])
    count = jnp.array([2],jnp.uint32)
    original = jnp.asarray(parents)
    current = PublishedBeamDepth(0,(original,),(2,),RankHistoryStore(world_size=1))
    tile_width = ((storage_width+127)//128)*128
    tiled = pallas_expand_state_rows(original,state_len=state_len,kernel_width=tile_width,interpret=True)
    table = pallas_expand_generator_rows(jnp.asarray(generators),state_len=state_len,
                                         kernel_width=tile_width,interpret=True)
    wire,material_error = pallas_materialize_final(tiled,table,jnp.asarray(requests),count,count,
                                                 state_len=state_len,interpret=True)
    _,targets = pallas_unpack_response(wire,state_len=state_len,interpret=True)
    prior = material_error[:1].at[0,0].set(int(late_error))
    common,coverage = make_final_coverage_agreement(SimpleNamespace(size=1),interpret=True)(
        targets,valid,count,prior)
    candidate,scatter_error = pallas_scatter_compact_final_responses(
        jnp.zeros_like(original),wire,count,state_len=state_len,prior_error=common,interpret=True)
    history = pallas_final_history_records(jnp.asarray(meta),targets,valid,interpret=True)
    published = commit_final_publication(
        current,frontier_by_rank=(candidate,),history_by_rank=(history,),target_counts=count,
        common_error=common[None],completed_work=(wire,material_error,coverage,scatter_error,history),
        move_count=moves,
    )
    np.testing.assert_array_equal(current.frontiers[0],parents)
    if late_error:
        assert published is current
        with pytest.raises(IndexError):
            current.history.read_entry(0,0,0)
    else:
        want = np.zeros_like(parents)
        want[:2,:state_len] = parents[:2,:state_len][:,::-1]
        np.testing.assert_array_equal(published.frontiers[0],want)
        assert published.history.read_entry(0,0,0) == HistoryEntry(0,0)
        assert published.history.read_entry(0,0,1) == HistoryEntry(1,0)
