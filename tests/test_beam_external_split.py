import numpy as np
import jax.numpy as jnp
import pytest
from tpu_beam_search import beam_external_sort as module


@pytest.mark.parametrize('n,count,mode', [(256, 0, 'mixed'), (256, 256, 'mixed'),
                                        (256, 129, 'local'), (256, 255, 'remote'),
                                        (512, 511, 'mixed')])
def test_external_split_preserves_records_order_counts_and_neutral_tail(n, count, mode):
    # Dropping parent high words, sorting by parent, or including invalid tails
    # must change this independent expected partition.
    assert hasattr(module, 'pallas_external_stream3_split')
    rank, world = 3, 8
    words = np.arange(8*n, dtype=np.uint32).reshape(8, n)
    words[5] += np.uint32(0x80000000)
    owners = np.arange(n, dtype=np.uint32) % world
    if mode == 'local':
        owners[:] = rank
    if mode == 'remote':
        owners[:] = 7
    control = np.zeros((1, 128), np.uint32)
    control[0, 0] = count
    actual = module.pallas_external_stream3_split(
        jnp.asarray(words), jnp.asarray(owners[None]), jnp.asarray(control),
        local_rank=rank, world_size=world, interpret=True)
    records = words.copy()
    records[7] = (rank << 16) | (owners << 8) | (words[7] & 255)
    local_ids = [i for i in range(count) if owners[i] == rank]
    remote_ids = sorted((i for i in range(count) if owners[i] != rank),
                        key=lambda i: (int(owners[i]), i))
    expected = []
    for ids in (local_ids, remote_ids):
        out = np.zeros_like(words)
        out[6] = np.uint32(0xffffffff)
        out[:, :len(ids)] = records[:, ids]
        expected.append(out)
    lc = np.zeros_like(control)
    lc[0, 0] = len(local_ids)
    counts = np.zeros_like(control)
    for i in remote_ids:
        counts[0, owners[i]] += 1
    offsets = np.zeros_like(control)
    offsets[0, 1:world+1] = np.cumsum(counts[0, :world])
    for got, want in zip(actual, (*expected, lc, counts, offsets), strict=True):
        np.testing.assert_array_equal(got, want)
