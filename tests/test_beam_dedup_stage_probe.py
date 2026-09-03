import numpy as np
import pytest

from benchmarks.beam_dedup_stage_probe import build_cases


def test_dedup_stage_probe_has_monotonic_real_pipeline_boundaries():
    assert [case['name'] for case in build_cases(interpret=True)] == [
        'initial', 'first_sort', 'uniqueness', 'second_sort', 'final_select']


@pytest.mark.parametrize('case', build_cases(interpret=True), ids=lambda case: case['name'])
def test_dedup_stage_probe_interpreter_matches_independent_numpy(case):
    actual = case['fn'](*case['args'])
    np.testing.assert_array_equal(np.asarray(actual), case['expected'])

