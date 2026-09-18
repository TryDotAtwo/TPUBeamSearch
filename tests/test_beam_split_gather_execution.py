from benchmarks.beam_split_gather_execution import expected
import numpy as np


def test_split_gather_oracle_covers_cross_tile_and_empty_intervals():
    data = np.arange(32 * 2048, dtype=np.uint32).reshape(32, 2048)
    ranges = np.zeros((3, 128), np.uint32)
    ranges[0, :4] = [1, 127, 255, 256]
    ranges[1, :4] = [128, 2, 1, 129]
    actual = expected(data, ranges, np.zeros((1, 128), np.uint32))
    assert [int(actual[p, 0, 0]) for p in range(4)] == [1, 127, 255, 256]
    assert np.array_equal(actual[4:], np.zeros((4, 32, 128), np.uint32))


def test_split_gather_oracle_preserves_prior_error_zeroing():
    data = np.arange(32 * 2048, dtype=np.uint32).reshape(32, 2048)
    ranges = np.zeros((3, 128), np.uint32)
    ranges[1, 0] = 128
    prior = np.zeros((1, 128), np.uint32)
    prior[0, 0] = 1
    assert not expected(data, ranges, prior).any()


def test_execution_gate_uses_per_program_source_abi():
    # pallas grid=(8,) selects the peer; payload remains one 32x2048 HBM tile.
    import inspect
    from benchmarks import beam_split_gather_execution as gate
    source = inspect.getsource(gate.run)
    assert "reshape(32, 2048)" in source
    assert "ranges = np.zeros((3, 128)" in source
