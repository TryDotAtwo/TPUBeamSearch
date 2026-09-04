import numpy as np
import pytest

from benchmarks.beam_previous_index_probe import build_cases


def test_previous_index_probe_has_one_control_and_two_equivalent_candidates():
    assert [case['name'] for case in build_cases(interpret=True)] == [
        'maximum_control', 'where_candidate', 'arithmetic_candidate']


@pytest.mark.parametrize('case', build_cases(interpret=True), ids=lambda case: case['name'])
def test_previous_index_probe_interpreter_matches_numpy(case):
    np.testing.assert_array_equal(np.asarray(case['fn'](*case['args'])), case['expected'])
