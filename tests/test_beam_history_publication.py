import pytest
from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore


def test_late_rank_failure_does_not_publish_any_rank_then_retry_succeeds():
    store = RankHistoryStore(world_size=3)
    a,b = HistoryEntry(0xffffffff00000001,7),HistoryEntry(4,0x10002)
    with pytest.raises(ValueError,match='duplicate'):
        store.append_all_rank_layer([[(0,a)],[],[(0,b),(0,b)]],target_counts=[1,0,2],depth=0)
    for rank in (0,2):
        with pytest.raises(IndexError):
            store.read_entry(rank,0,0)
    store.append_all_rank_layer([[(0,a)],[],[(1,b),(0,a)]],target_counts=[1,0,2],depth=0)
    assert store.read_entry(0,0,0) == a
    assert store.read_entry(2,0,0) == a
    assert store.read_entry(2,0,1) == b


def test_stale_depth_and_missing_rank_cannot_advance_history():
    store = RankHistoryStore(world_size=2)
    a = HistoryEntry(3,5)
    store.append_all_rank_layer([[(0,a)],[]],target_counts=[1,0],depth=0)
    with pytest.raises(ValueError,match='depth'):
        store.append_all_rank_layer([[],[]],target_counts=[0,0],depth=0)
    with pytest.raises(ValueError,match='rank'):
        store.append_all_rank_layer([[]],target_counts=[0],depth=1)
    store.append_all_rank_layer([[],[(0,a)]],target_counts=[0,1],depth=1)
    assert store.read_entry(0,0,0) == a
    assert store.read_entry(1,1,0) == a


def test_generator_failure_does_not_publish_earlier_valid_rank():
    store = RankHistoryStore(world_size=2)
    a = HistoryEntry(3,5)
    def failed_copy():
        yield 0,a
        raise RuntimeError('incomplete host transfer')
    with pytest.raises(RuntimeError,match='incomplete'):
        store.append_all_rank_layer([[(0,a)],failed_copy()],target_counts=[1,1],depth=0)
    with pytest.raises(IndexError):
        store.read_entry(0,0,0)
