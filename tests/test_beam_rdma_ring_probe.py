import numpy as np

from benchmarks.beam_rdma_ring_probe import evaluate_right_permute


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
