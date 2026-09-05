import importlib.util
import pytest


@pytest.mark.parametrize('used,busy,current,amount,expected', [
    ((3,0),(False,False),0,5,0),
    ((7,0),(False,False),0,2,1),
    ((0,3),(True,False),0,5,1),
    ((7,7),(False,False),0,2,None),
    ((0,0),(True,True),0,1,None),
])
def test_reserve_group_prefers_current_but_never_busy_or_overflow(used,busy,current,amount,expected):
    assert importlib.util.find_spec('tpu_beam_search.beam_collector') is not None
    from tpu_beam_search.beam_collector import reserve_group
    before = (used,busy)
    result = reserve_group(capacity=8,clean=used,dirty=(0,0),processing=busy,
                           current=current,amount=amount)
    assert (used,busy) == before
    assert result.buffer == expected
    assert result.fatal_overflow == (expected is None)
    if expected is not None:
        assert result.offset == used[expected]
        assert result.dirty[expected] == amount
        assert result.dirty[1-expected] == 0
    else:
        assert result.dirty == (0,0)


def test_empty_arrival_does_not_overflow_full_busy_buffers():
    assert importlib.util.find_spec('tpu_beam_search.beam_collector') is not None
    from tpu_beam_search.beam_collector import reserve_group
    result = reserve_group(capacity=8,clean=(4,4),dirty=(4,4),
                           processing=(True,True),current=0,amount=0)
    assert result.buffer is None
    assert not result.fatal_overflow
    assert result.dirty == (4,4)
