import numpy as np

from benchmarks.beam_rdma_ring_probe import evaluate_epoch_ring, evaluate_right_permute


def test_right_permute_report_detects_exact_ring_and_one_wrong_word():
    source = np.array([
        [[10, 11]], [[20, 21]], [[30, 31]], [[40, 41]],
    ], dtype=np.uint32)
    rotated = np.array([
        [[40, 41]], [[10, 11]], [[20, 21]], [[30, 31]],
    ], dtype=np.uint32)
    exact = evaluate_right_permute(rotated, source)
    assert exact['exact']
    assert exact['mismatched_elements'] == 0

    corrupted = rotated.copy()
    corrupted[2, 0, 1] ^= 1
    mismatch = evaluate_right_permute(corrupted, source)
    assert not mismatch['exact']
    assert mismatch['mismatched_elements'] == 1


def test_right_permute_report_rejects_structure_mismatch():
    source = np.zeros((8, 8, 128), np.uint32)
    result = evaluate_right_permute(np.zeros((8, 128), np.uint32), source)
    assert not result['exact']
    assert result['structure_mismatch']


def test_epoch_ring_oracle_rotates_active_epochs_and_zeros_inactive_epochs():
    source = np.array([
        [[[10]], [[11]], [[12]]],
        [[[20]], [[21]], [[22]]],
        [[[30]], [[31]], [[32]]],
        [[[40]], [[41]], [[42]]],
    ], dtype=np.uint32)
    actual = np.array([
        [[[40]], [[0]], [[42]]],
        [[[10]], [[0]], [[12]]],
        [[[20]], [[0]], [[22]]],
        [[[30]], [[0]], [[32]]],
    ], dtype=np.uint32)
    report = evaluate_epoch_ring(actual, source, active_epochs=(True, False, True))
    assert report['exact']
    assert report['mismatched_elements'] == 0
