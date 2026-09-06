import numpy as np


def test_probe_expected_errors_match_independent_target_sets():
    from benchmarks.beam_final_coverage_probe import fixtures
    cases = list(fixtures())
    assert len(cases) == 7
    for name,(targets,valid,counts,prior),expected in cases:
        failures = []
        for rank in range(8):
            live = targets[rank,0,valid[rank,0] != 0]
            n = int(counts[rank,0])
            failures.append(not np.array_equal(np.sort(live),np.arange(n,dtype=np.uint32)))
        assert int(any(failures) or prior.any()) == expected, name
        if expected and name != 'prior_error':
            assert np.flatnonzero(failures).tolist() == [0 if name == 'extra' else 4]
        if name == 'prior_error':
            assert not any(failures)
            assert np.flatnonzero(prior[:,0,0]).tolist() == [6]
