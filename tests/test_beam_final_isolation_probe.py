import numpy as np
import pytest


def test_report_gate_requires_both_counts_and_eight_exact_devices():
    from benchmarks.beam_final_isolation_probe import report_exact
    rows = [dict(count=c,mismatches=[0]*8) for c in (0,2)]
    assert report_exact(rows)
    assert not report_exact(rows[:1])
    assert not report_exact([rows[0],rows[0]])
    assert not report_exact([rows[0],dict(count=2,mismatches=[0]*7)])
    assert not report_exact([rows[0],dict(count=2,mismatches=[0]*7+[1])])


@pytest.mark.parametrize('mode',['dma','cast','gather_1d','gather_2d','packing','production'])
@pytest.mark.parametrize('count',[0,2])
def test_isolated_steps_match_literal_bytes(mode,count):
    from benchmarks.beam_final_isolation_probe import fixture, run_probe
    inputs,expected = fixture(mode,count)
    actual = run_probe(mode,*inputs,interpret=True)
    np.testing.assert_array_equal(actual,expected)
