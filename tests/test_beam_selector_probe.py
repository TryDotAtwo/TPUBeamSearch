import numpy as np
import pytest

from benchmarks.beam_selector_probe import build_cases


def test_selector_probe_covers_shared_survivor_forms():
    names = {case['name'] for case in build_cases(interpret=True)}
    assert names == {
        'where_broadcast_bool',
        'where_full_bool',
        'arithmetic_broadcast_bool',
        'arithmetic_full_bool',
        'arithmetic_u32_mask',
    }


@pytest.mark.parametrize('case', build_cases(interpret=True), ids=lambda case: case['name'])
def test_selector_probe_interpreter_is_exact(case):
    actual = case['fn'](*case['args'])
    np.testing.assert_array_equal(np.asarray(actual), case['expected'])

