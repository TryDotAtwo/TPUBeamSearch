import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.beam_stream3 import pallas_stream3_split
from tpu_beam_search.beam_types import pack_candidates


@pytest.mark.parametrize('count,world,rank', [(7, 8, 3), (7, 1, 0), (0, 8, 3)])
def test_stream3_split_compacts_local_and_groups_remote_stably(count, world, rank):
    hashes = [90, 10, 80, 20, 70, 30, 60]
    parents = [2**40 + i for i in range(7)]
    scores = list(range(7))
    moves = [9, 1, 8, 2, 7, 3, 6]
    words = pack_candidates(hashes, parents, scores, moves, capacity=128)
    owners = np.array([[3, 1, 3, 0, 7, 1, 0] + [0] * 121], np.uint32) % world
    local, remote, local_count, send_count, send_offset = pallas_stream3_split(
        jnp.array(words), jnp.array(owners), jnp.array([count], np.uint32),
        local_rank=rank, world_size=world, interpret=True)
    owners_valid = owners[0, :count]
    local_ids = [i for i in range(count) if owners_valid[i] == rank]
    remote_ids = [i for peer in range(world) if peer != rank
                  for i in range(count) if owners_valid[i] == peer]

    def expected(ids):
        value = words[:, ids].copy()
        if ids:
            value[7] = np.uint32(rank << 16) | (owners[0, ids] << np.uint32(8)) | value[7]
        return value

    assert int(local_count[0]) == len(local_ids)
    np.testing.assert_array_equal(local[:, :len(local_ids)], expected(local_ids))
    np.testing.assert_array_equal(remote[:, :len(remote_ids)], expected(remote_ids))
    neutral = np.zeros((8, 128 - len(local_ids)), np.uint32)
    neutral[6] = 0xffffffff
    np.testing.assert_array_equal(local[:, len(local_ids):], neutral)
    counts = np.array([0 if peer == rank else np.count_nonzero(owners_valid == peer)
                       for peer in range(world)], np.uint32)
    np.testing.assert_array_equal(send_count[0, :world], counts)
    np.testing.assert_array_equal(send_offset[0, :world + 1], np.r_[0, np.cumsum(counts)])
    assert np.all(send_count[0, world:] == 0)
    assert np.all(send_offset[0, world + 1:] == 0)


@pytest.mark.parametrize('kwargs', [
    dict(local_rank=8, world_size=8), dict(local_rank=0, world_size=0),
    dict(local_rank=0, world_size=257),
])
def test_stream3_split_rejects_invalid_static_topology(kwargs):
    words = jnp.zeros((8, 128), jnp.uint32)
    owners = jnp.zeros((1, 128), jnp.uint32)
    with pytest.raises(ValueError):
        pallas_stream3_split(words, owners, jnp.zeros((1,), jnp.uint32),
                             interpret=True, **kwargs)
