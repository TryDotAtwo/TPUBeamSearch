import numpy as np
import pytest

from benchmarks.beam_compare_exchange_probe import build_cases


def test_compare_exchange_probe_covers_real_first_comparator_boundaries():
    assert [case['name'] for case in build_cases(interpret=True)] == [
        'partner_gather', 'swap_predicate', 'select_broadcast',
        'select_full', 'select_rowwise', 'select_arithmetic']


@pytest.mark.parametrize('case', build_cases(interpret=True), ids=lambda case: case['name'])
def test_compare_exchange_probe_interpreter_matches_numpy(case):
    np.testing.assert_array_equal(np.asarray(case['fn'](*case['args'])), case['expected'])

