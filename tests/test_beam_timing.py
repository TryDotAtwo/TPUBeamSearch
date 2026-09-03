import numpy as np

from benchmarks.beam_primitive_bundle import measure_interleaved


def test_timing_alternates_order_and_excludes_warmups_from_samples():
    calls = []
    def first():
        calls.append('A')
        return np.array([1], dtype=np.uint32)
    def second():
        calls.append('B')
        return np.array([2], dtype=np.uint32)
    result = measure_interleaved({'A': first, 'B': second}, warmup=2, repeats=4)
    assert calls == ['A', 'B', 'A', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A']
    assert len(result['A']['samples_ms']) == 4
    assert len(result['B']['samples_ms']) == 4
    assert result['A']['median_ms'] >= 0
