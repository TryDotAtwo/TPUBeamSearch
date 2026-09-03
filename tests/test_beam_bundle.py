import numpy as np

from benchmarks.beam_primitive_bundle import build_cases, compare_outputs


def test_bundle_has_independent_oracles_and_rejects_one_wrong_bit():
    cases = build_cases(interpret=True)
    assert {'pack_serial_b2', 'pack_pipeline_b3', 'route_8_7',
            'hash_goal_120_24', 'hash_goal_150_30', 'dedup_stream3_128',
            'dedup_stream4_256'} <= {c['name'] for c in cases}
    case = next(c for c in cases if c['name'] == 'route_8_7')
    out = case['fn'](*case['args'])
    assert compare_outputs(out, case['expected'])['exact']
    corrupted = np.asarray(out).copy()
    corrupted[0, 0] ^= 1
    result = compare_outputs(corrupted, case['expected'])
    assert not result['exact']
    assert result['mismatched_elements'] == 1


def test_split_cases_have_independent_complete_buffer_oracles():
    cases = [c for c in build_cases(interpret=True) if c['name'].startswith('split_')]
    assert len(cases) == 2
    for case in cases:
        assert compare_outputs(case['fn'](*case['args']), case['expected'])['exact']
