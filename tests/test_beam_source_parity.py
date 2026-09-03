"""Optional compiled original-source differential gate, not a Python mirror.

Set BEAM_SOURCE_ORACLE to the adapter executable built against D:/100XH100.
Skipping this test means that source differential evidence was not collected.
"""
import os
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.beam_stream2 import pallas_hash_goal
from tpu_beam_search.beam_hash import pallas_route_hashes
from tpu_beam_search.beam_dedup import pallas_threshold_dedup
from tpu_beam_search.beam_types import pack_candidates


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'), reason='original C++ oracle not configured')
def test_original_cpp_hash_goal_matches_pallas():
    rng = np.random.default_rng(9341)
    parents = np.zeros((3, 128), np.uint8)
    parents[:, :120] = rng.integers(0, 12, (3, 120), dtype=np.uint8)
    generators = np.tile(np.arange(128, dtype=np.int32), (24, 1))
    for m in range(24):
        generators[m, :120] = rng.permutation(120)
    central = parents[1, generators[23]].copy()
    table = rng.integers(0, 2**32, (4, 128, 12), dtype=np.uint32)
    table[:, 120:] = 0
    tokens = [3, 24, 12, *parents.ravel(), *generators.ravel(), *central,
              *table.transpose(1, 2, 0).ravel()]
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE']],
                         input=' '.join(map(str, tokens)), text=True,
                         capture_output=True, check=True, timeout=30)
    expected = np.array([list(map(int, line.split())) for line in raw.stdout.splitlines()], np.uint32)
    assert expected.shape == (72, 7)
    assert np.count_nonzero(expected[:, 4]) == 1
    actual_hash, actual_goal, valid = pallas_hash_goal(
        jnp.array(parents), jnp.array(generators), jnp.array(central),
        jnp.array(table.reshape(4, -1)), jnp.array([3], jnp.uint32), interpret=True)
    np.testing.assert_array_equal(actual_hash[:, :72], expected[:, :4].T)
    np.testing.assert_array_equal(actual_goal[0, :72], expected[:, 4])
    assert np.count_nonzero(valid) == 72
    routes = pallas_route_hashes(actual_hash, world_size=8, shard_count=7, interpret=True)
    np.testing.assert_array_equal(routes[:, :72], expected[:, 5:7].T)


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'), reason='original C++ oracle not configured')
def test_original_cpp_stream4_matches_pallas():
    rng = np.random.default_rng(335)
    hashes = [int(x) * (2**97 + 17) for x in rng.integers(0, 11, 127)]
    parents = [int(x) + 2**40 for x in rng.integers(0, 8, 127)]
    scores = rng.integers(0, 5, 127).tolist()
    routes = rng.integers(0, 2**32, 127, dtype=np.uint32).tolist()
    words = pack_candidates(hashes, parents, scores, routes, capacity=128)
    tokens = [127, 3, *words[:, :127].T.ravel()]
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE'], 'dedup'],
                         input=' '.join(map(str, tokens)), text=True,
                         capture_output=True, check=True, timeout=30)
    expected = np.array([list(map(int, line.split())) for line in raw.stdout.splitlines()], np.uint32).T
    out, count = pallas_threshold_dedup(jnp.array(words), jnp.zeros((1, 128), jnp.uint32),
        jnp.array([127], jnp.uint32), jnp.array([3], jnp.uint32), mode='stream4', interpret=True)
    assert int(count[0]) == expected.shape[1]
    np.testing.assert_array_equal(out[:, :expected.shape[1]], expected)
