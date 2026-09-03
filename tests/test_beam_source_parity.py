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


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'), reason='original C++ oracle not configured')
@pytest.mark.parametrize('world,rank,threshold,count', [(8, 3, 3, 127), (1, 0, 0xffffffff, 127), (8, 3, 0, 0)])
def test_original_cpp_stream3_payload_ties_and_routing(world, rank, threshold, count):
    # A change from payload tie-break to parent or compacted position must fail.
    rng = np.random.default_rng(971)
    hashes = [int(x) * (2**99 + 19) for x in rng.integers(0, 29, 127)]
    parents = [2**48 + i for i in range(127)]
    scores = rng.integers(0, 5, 127).tolist()
    moves = (np.arange(127, dtype=np.uint32) % 24).tolist()
    payload = np.arange(127, 0, -1, dtype=np.uint32)
    # Equal scores/hash, opposing parent and payload order, different moves.
    hashes[:2] = [2**127 + 7] * 2
    scores[:2] = [0, 0]
    words = pack_candidates(hashes, parents, scores, moves, capacity=128)
    records = np.concatenate((words[:, :count], payload[None, :count]), axis=0)
    tokens = [count, threshold, rank, world, *records.T.ravel()]
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE'], 'stream3'],
        input=' '.join(map(str, tokens)), text=True, capture_output=True, timeout=30)
    assert raw.returncode == 0, raw.stderr
    lines = raw.stdout.splitlines()
    assert lines[0] == 'STREAM3'
    local_n, remote_n = map(int, lines[1].split())
    send_count = np.array(list(map(int, lines[2].split())), np.uint32)
    offsets = np.array(list(map(int, lines[3].split())), np.uint32)
    expected = np.array([list(map(int, line.split())) for line in lines[4:]], np.uint32).reshape(-1, 8)
    assert expected.shape == (local_n + remote_n, 8)
    padded_payload = np.pad(payload, (0, 1))[None, :]
    out, valid_count = pallas_threshold_dedup(jnp.array(words), jnp.array(padded_payload),
        jnp.array([count], jnp.uint32), jnp.array([threshold], jnp.uint32), mode='stream3', interpret=True)
    size = int(valid_count[0])
    routed = pallas_route_hashes(out[:4], world_size=world, shard_count=7, interpret=True)
    actual = np.array(out[:, :size]).T.copy()
    owners = np.asarray(routed[0, :size])
    actual[:, 7] = np.uint32(rank << 16) | (owners << np.uint32(8)) | actual[:, 7]
    # Host partition is a test adapter only, NOT a TPU split implementation.
    local = actual[owners == rank]
    remote = np.concatenate([actual[owners == peer] for peer in range(world) if peer != rank], axis=0) if world > 1 else actual[:0]
    np.testing.assert_array_equal(np.concatenate((local, remote)), expected)
    counts = np.array([np.count_nonzero(owners == peer) if peer != rank else 0 for peer in range(world)], np.uint32)
    np.testing.assert_array_equal(send_count, counts)
    np.testing.assert_array_equal(offsets, np.concatenate(([0], np.cumsum(counts))))
    assert local_n == len(local) and remote_n == len(remote)
    if count:
        winner = actual[(actual[:, 0] == 7) & (actual[:, 3] == 0x80000000)]
        assert winner.shape == (1, 8)
        assert int(winner[0, 4]) == 1  # parent 2**48 + 1, not smaller parent
        assert int(winner[0, 7] & 255) == 1
