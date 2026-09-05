import jax.numpy as jnp
import numpy as np

from tpu_beam_search import beam_collector as module
from test_beam_hash import oracle


def test_hash_partition_uses_shard_salt_not_owner_and_preserves_metadata():
    assert hasattr(module, 'pallas_collector_hash_partition')
    n, valid, shard_count = 256, 213, 3
    words = np.random.default_rng(128).integers(0, 2**32, (8,n), dtype=np.uint32)
    ids, owners = [], []
    for i in range(valid):
        lo = int(words[0,i]) | (int(words[1,i]) << 32)
        hi = int(words[2,i]) | (int(words[3,i]) << 32)
        ids.append(oracle(lo,hi,0x13198a2e03707344) % shard_count)
        owners.append(oracle(lo,hi,0x243f6a8885a308d3) % shard_count)
    assert ids != owners
    count = np.zeros((1,128),np.uint32)
    count[0,0] = valid
    actual, counts, offsets = module.pallas_collector_hash_partition(
        jnp.asarray(words),jnp.asarray(count),shard_count=shard_count,interpret=True)
    order = sorted(range(valid),key=lambda i:(ids[i],i))
    expected = np.zeros_like(words)
    expected[6] = np.uint32(0xffffffff)
    expected[:,:valid] = words[:,order]
    np.testing.assert_array_equal(actual,expected)
    totals = np.zeros((1,128),np.uint32)
    totals[0,:shard_count] = np.bincount(ids,minlength=shard_count)
    starts = np.zeros_like(totals)
    starts[0,1:shard_count+1] = np.cumsum(totals[0,:shard_count])
    np.testing.assert_array_equal(counts,totals)
    np.testing.assert_array_equal(offsets,starts)
