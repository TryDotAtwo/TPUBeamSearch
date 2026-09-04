import numpy as np

from benchmarks.beam_rdma_ring_probe import (
    call_compiled,
    evaluate_epoch_ring,
    evaluate_integrated_stream3_exchange,
    evaluate_right_permute,
    evaluate_variable_exchange,
)


def test_call_compiled_splats_multi_input_placement():
    calls = []

    def executable(*args):
        calls.append(args)
        return args

    assert call_compiled(executable, ('counts', 'payload')) == ('counts', 'payload')
    assert calls == [('counts', 'payload')]
    assert call_compiled(executable, 'single') == ('single',)
    assert calls[-1] == ('single',)


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


def test_variable_exchange_oracle_routes_each_offset_and_preserves_zero_count():
    neutral = np.zeros((3, 2, 8, 2), np.uint32)
    neutral[:, :, 6, :] = np.uint32(0xffffffff)
    sends = neutral.copy()
    sends[0, 0, 0, 0] = 10  # rank 0 -> rank 1
    sends[1, 1, 0, 0] = 21  # rank 1 -> rank 0 (offset 2)
    sends[2, 0, 0, :2] = [30, 31]  # rank 2 -> rank 0
    counts = np.array([[1, 0], [0, 1], [2, 0]], np.uint32)

    received = neutral.copy()
    received[0, 0] = sends[2, 0]
    received[0, 1] = sends[1, 1]
    received[1, 0] = sends[0, 0]
    received_counts = np.array([[2, 1], [1, 0], [0, 0]], np.uint32)
    report = evaluate_variable_exchange(received, received_counts, sends, counts)
    assert report['exact']
    assert report['mismatched_elements'] == 0

    received[1, 0, 0, 0] ^= 1
    assert not evaluate_variable_exchange(
        received, received_counts, sends, counts)['exact']


def test_integrated_stream3_oracle_checks_local_remote_routes_and_neutral_tails():
    neutral = np.zeros((8, 2), np.uint32)
    neutral[6] = 0xffffffff
    words = np.broadcast_to(neutral, (2, 8, 2)).copy()
    words[0, :, 0] = [10, 0, 0, 0, 0, 0, 1, 3]
    words[0, :, 1] = [11, 0, 0, 0, 0, 0, 2, 4]
    words[1, :, 0] = [20, 0, 0, 0, 0, 0, 3, 5]
    words[1, :, 1] = [21, 0, 0, 0, 0, 0, 4, 6]
    owners = np.array([[0, 1], [0, 1]], np.uint32)
    counts = np.array([2, 2], np.uint32)

    local = np.broadcast_to(neutral, (2, 8, 2)).copy()
    local[0, :, 0] = words[0, :, 0]
    local[0, 7, 0] = 3
    local[1, :, 0] = words[1, :, 1]
    local[1, 7, 0] = (1 << 16) | (1 << 8) | 6
    local_counts = np.array([1, 1], np.uint32)
    received = np.broadcast_to(neutral, (2, 1, 8, 2)).copy()
    received[0, 0, :, 0] = words[1, :, 0]
    received[0, 0, 7, 0] = (1 << 16) | 5
    received[1, 0, :, 0] = words[0, :, 1]
    received[1, 0, 7, 0] = (1 << 8) | 4
    received_counts = np.ones((2, 1), np.uint32)
    wire = received[::-1].copy()
    wire_counts = np.ones((2, 1), np.uint32)

    report = evaluate_integrated_stream3_exchange(
        local, local_counts, received, received_counts,
        words, owners, counts, wire=wire, wire_counts=wire_counts)
    assert report['exact']
    assert report['mismatched_elements'] == 0

    received[0, 0, 7, 0] ^= 1
    assert not evaluate_integrated_stream3_exchange(
        local, local_counts, received, received_counts,
        words, owners, counts, wire=wire,
        wire_counts=wire_counts)['exact']
