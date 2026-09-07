import numpy as np
import pytest


@pytest.mark.parametrize('count',[0,1,127,128,129])
def test_scatter_fixture_has_literal_canaries_and_target_tail(count):
    from benchmarks.beam_final_scatter_fixture import make_case
    case = make_case(count)
    assert case['expected_error'] == 0
    parents,perm,requests,n,frontier,wire = case['inputs']
    assert int(n[0]) == count
    targets = (np.arange(count)*3)%256
    np.testing.assert_array_equal(requests[2,:count],targets)
    expected = frontier.copy()
    for i,t in enumerate(targets):
        np.testing.assert_array_equal(wire[i,:120],parents[i%7,perm[0,:120]])
        assert int.from_bytes(wire[i,120:124].tobytes(),'little') == t
        expected[t,:120] = wire[i,:120]
        expected[t,120:] = 0
    np.testing.assert_array_equal(case['expected'],expected)


@pytest.mark.parametrize('failure',['count_overflow','target_overflow'])
def test_rejected_scatter_fixture_preserves_entire_frontier(failure):
    from benchmarks.beam_final_scatter_fixture import make_case
    case = make_case(129,failure=failure)
    assert case['expected_error'] == 1
    np.testing.assert_array_equal(case['expected'],case['inputs'][4])
