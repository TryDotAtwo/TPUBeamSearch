import numpy as np


def test_epoch_probe_covers_every_requester_and_carries_expected_state():
    from benchmarks.beam_s5_epoch_probe import fixtures
    initial,steps = fixtures()
    assert len(steps) == 20
    assert np.count_nonzero(initial[0]) == 1
    assert initial[0][5,0,5] == 9
    updates = 0
    for index,(force,expected) in enumerate(steps):
        phase = index%10
        updates += int(phase != 0)
        np.testing.assert_array_equal(expected[3][:,1,0],updates)
        np.testing.assert_array_equal(expected[2][:,0,0],updates%2)
        assert np.count_nonzero(force) == (0 if phase == 0 else 8 if phase == 9 else 1)
        if 1 <= phase <= 8:
            assert force[phase-1,0] == 1
