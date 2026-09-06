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
