import numpy as np
import pytest


def test_chunking_preserves_device_partition_and_reconstructs_global_row_order():
    import benchmarks.artgor_prefix_shape as shape
    states = np.arange(24).reshape(12,2)
    seen = []
    def operation(chunk):
        seen.append(chunk[:,0].tolist())
        return chunk * 3 + 1
    actual = shape.chunked_host(states, operation, devices=3, chunk_rows=2)
    assert seen == [[0,2,8,10,16,18], [4,6,12,14,20,22]]
    np.testing.assert_array_equal(actual, states*3+1)


def test_chunking_rejects_incomplete_device_or_chunk_partitions():
    import benchmarks.artgor_prefix_shape as shape
    with pytest.raises(ValueError, match='partition'):
        shape.chunked_host(np.zeros((11,2)), lambda x:x, devices=3, chunk_rows=2)
    with pytest.raises(ValueError, match='partition'):
        shape.chunked_host(np.zeros((12,2)), lambda x:x, devices=3, chunk_rows=3)
