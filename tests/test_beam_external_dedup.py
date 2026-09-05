import jax.numpy as jnp
import numpy as np
import os
import subprocess
import pytest

from tpu_beam_search.beam_external_sort import pallas_mark_sorted_unique
from tpu_beam_search.beam_external_sort import pallas_external_stream3_dedup


def test_unique_marker_compares_predecessor_across_tiles_and_keeps_hash_zero():
    data = np.zeros((11, 256), np.uint32)
    data[0] = np.arange(256, dtype=np.uint32)
    data[0, 128] = data[0, 127]
    data[0, 200:] = 0
    data[9, :200] = 1
    data[10] = np.arange(256, dtype=np.uint32)[::-1]
    expected = data.copy()
    expected[9, 128] = 0
    expected[10] = np.arange(256, dtype=np.uint32)
    actual = pallas_mark_sorted_unique(jnp.asarray(data), interpret=True)
    np.testing.assert_array_equal(actual, expected)


def test_unique_marker_all_duplicate_and_empty():
    for valid in (0, 1):
        data = np.zeros((11, 256), np.uint32)
        data[:4] = np.uint32(0xffffffff)
        data[9] = valid
        expected = data.copy()
        expected[9] = 0
        expected[9, 0] = valid
        expected[10] = np.arange(256, dtype=np.uint32)
        np.testing.assert_array_equal(
            pallas_mark_sorted_unique(jnp.asarray(data), interpret=True), expected)


def test_external_stream3_threshold_winner_compaction_and_neutral_tail():
    n = 256
    words = np.zeros((8, n), np.uint32)
    words[0] = np.arange(n, dtype=np.uint32)
    words[4] = np.arange(n, dtype=np.uint32) + 1000
    words[5] = np.uint32(0x80000000)
    words[6] = 10
    words[0, 129] = words[0, 3]
    words[6, 3] = 11  # threshold rejects this record, but not its duplicate
    words[0, 130] = words[0, 4]
    words[0, 131] = words[0, 4]
    words[6, 131] = 9  # score wins before payload
    payload = np.arange(n, dtype=np.uint32)[None, :]
    payload[0, 130] = 0
    count, threshold = 201, 10
    order = sorted((i for i in range(count) if words[6, i] <= threshold),
                   key=lambda i: (tuple(int(words[p, i]) for p in (3, 2, 1, 0)),
                                  int(words[6, i]), int(payload[0, i])))
    keep, seen = [], set()
    for i in order:
        key = tuple(words[:4, i])
        if key not in seen:
            keep.append(i)
            seen.add(key)
    expected = np.zeros_like(words)
    expected[6] = np.uint32(0xffffffff)
    expected[:, :len(keep)] = words[:, keep]
    actual, actual_count = pallas_external_stream3_dedup(
        jnp.asarray(words), jnp.asarray(payload),
        jnp.array([count], jnp.uint32), jnp.array([threshold], jnp.uint32),
        interpret=True)
    np.testing.assert_array_equal(actual, expected)
    expected_count = np.zeros((1, 128), np.uint32)
    expected_count[0, 0] = len(keep)
    np.testing.assert_array_equal(actual_count, expected_count)


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'),
                    reason='original C++ oracle not configured')
def test_external_dedup_matches_original_stream3_cpp_at_256():
    rng = np.random.default_rng(6581)
    words = np.zeros((8, 256), np.uint32)
    words[0] = rng.integers(0, 70, 256, dtype=np.uint32)
    words[3] = np.uint32(0x80000000)
    words[4] = np.arange(256, dtype=np.uint32)
    words[5] = np.uint32(0x10000)
    words[6] = rng.integers(0, 8, 256, dtype=np.uint32)
    words[7] = np.arange(256, dtype=np.uint32) % 24
    payload = np.arange(255, -1, -1, dtype=np.uint32)[None, :]
    records = np.concatenate((words, payload))
    # world=1 leaves routing unchanged, isolating the original S3 winner order.
    tokens = [256, 5, 0, 1, *records.T.ravel()]
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE'], 'stream3'],
        input=' '.join(map(str, tokens)), text=True, capture_output=True,
        check=True, timeout=30)
    lines = raw.stdout.splitlines()
    expected = np.array([list(map(int, row.split())) for row in lines[4:]],
                        dtype=np.uint32).reshape(-1, 8)
    actual, count = pallas_external_stream3_dedup(
        jnp.asarray(words), jnp.asarray(payload), jnp.array([256], jnp.uint32),
        jnp.array([5], jnp.uint32), interpret=True)
    assert int(count[0, 0]) == len(expected)
    np.testing.assert_array_equal(actual[:, :len(expected)].T, expected)
