import pytest


def test_history_follows_source_rank_not_balanced_owner_and_preserves_parent64():
    from tpu_beam_search.beam_history import HistoryEntry, reconstruct_history
    calls = []
    high_parent = (1 << 32) + 9
    records = {(6,1,high_parent): HistoryEntry(4,(2<<16)|(7<<8)|11),
               (2,0,4): HistoryEntry(0,(5<<16)|(3<<8)|8)}
    def read(rank,layer,index):
        calls.append((rank,layer,index))
        return records[rank,layer,index]
    path = reconstruct_history(HistoryEntry(high_parent,(6<<16)|(1<<8)|23),
        depth=3,world_size=8,move_count=24,read_entry=read)
    assert path.moves == (8,11,23)
    assert path.parent_indices == (0,4,high_parent)
    assert calls == [(6,1,high_parent),(2,0,4)]


def test_history_rejects_invalid_rank_before_reading():
    from tpu_beam_search.beam_history import HistoryEntry, reconstruct_history
    def read(*args):
        pytest.fail('invalid rank reached history service')
    with pytest.raises(ValueError,match='rank'):
        reconstruct_history(HistoryEntry(0,(8<<16)|1),depth=2,
            world_size=8,move_count=24,read_entry=read)


@pytest.mark.parametrize('depth,expected',[(0,()),(1,(7,))])
def test_root_and_empty_paths_do_not_query_history(depth,expected):
    from tpu_beam_search.beam_history import HistoryEntry, reconstruct_history
    def read(*args):
        pytest.fail('root must not query previous history')
    path = reconstruct_history(HistoryEntry(0,7),depth=depth,world_size=8,
        move_count=24,read_entry=read)
    assert path.moves == expected


def test_invalid_move_is_rejected_before_history_lookup():
    from tpu_beam_search.beam_history import HistoryEntry, reconstruct_history
    with pytest.raises(ValueError,match='move'):
        reconstruct_history(HistoryEntry(0,24),depth=2,world_size=8,
            move_count=24,read_entry=lambda *args: pytest.fail('unexpected read'))


@pytest.mark.parametrize('parent,route',[(-1,0),(1<<64,0),(0,-1),(0,1<<32)])
def test_history_entry_rejects_out_of_range_words(parent,route):
    from tpu_beam_search.beam_history import HistoryEntry
    with pytest.raises(ValueError):
        HistoryEntry(parent,route)


def test_store_reorders_target_slots_and_rejects_duplicate_missing_atomically():
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore
    store = RankHistoryStore(world_size=2)
    a,b = HistoryEntry(9,3),HistoryEntry(4,7)
    with pytest.raises(ValueError,match='duplicate'):
        store.append_rank_layer(0,[(0,a),(0,b)],target_count=2)
    with pytest.raises(ValueError,match='missing'):
        store.append_rank_layer(0,[(0,a)],target_count=2)
    store.append_rank_layer(0,[(1,b),(0,a)],target_count=2)
    store.append_rank_layer(1,[],target_count=0)
    assert store.read_entry(0,0,0) == a
    assert store.read_entry(0,0,1) == b
    with pytest.raises(IndexError):
        store.read_entry(1,0,0)
    with pytest.raises(IndexError):
        store.read_entry(0,0,-1)


def test_rank_store_reconstruction_crosses_balanced_frontiers():
    from tpu_beam_search.beam_history import HistoryEntry, RankHistoryStore, reconstruct_history
    store = RankHistoryStore(world_size=3)
    for rank in range(3):
        store.append_rank_layer(rank,[(0,HistoryEntry(0,rank+1))],target_count=1)
    for rank in range(3):
        # This record resides on rank, but refers to another source rank.
        store.append_rank_layer(rank,[(0,HistoryEntry(0,(((rank+1)%3)<<16)|4))],target_count=1)
    path = reconstruct_history(HistoryEntry(0,(1<<16)|5),depth=3,
        world_size=3,move_count=6,read_entry=store.read_entry)
    assert path.moves == (3,4,5)
