import numpy as np
import pytest
from benchmarks.beam_final_scatter_fixture import make_case


@pytest.mark.parametrize('mode',['scatter','integrated'])
def test_probe_chain_matches_fixture(mode):
    from benchmarks.beam_final_scatter_probe import run_case
    case = make_case(1)
    actual,errors = run_case(mode,*case['inputs'],interpret=True)
    np.testing.assert_array_equal(actual,case['expected'])
    assert int(errors[0,0]) == 0


@pytest.mark.parametrize('failure',['count_overflow','target_overflow'])
def test_scatter_probe_rejects_whole_invalid_batch(failure):
    from benchmarks.beam_final_scatter_probe import run_case
    case = make_case(129,failure=failure)
    actual,errors = run_case('scatter',*case['inputs'],interpret=True)
    np.testing.assert_array_equal(actual,case['expected'])
    assert int(errors[0,0]) == case['expected_error']
