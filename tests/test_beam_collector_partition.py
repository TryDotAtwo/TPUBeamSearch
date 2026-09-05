import numpy as np
import jax.numpy as jnp
import pytest
from tpu_beam_search import beam_collector as module


@pytest.mark.parametrize('valid,shards,all_same',[(201,4,False),(0,3,False),(256,3,True)])
def test_partition_keeps_stable_shard_groups_and_excludes_padding(valid,shards,all_same):
    assert hasattr(module,'pallas_collector_partition')
    n = 256
    words = np.arange(8*n,dtype=np.uint32).reshape(8,n)
    ids = (np.arange(n,dtype=np.uint32)*3)%shards
    if all_same:
        ids[:] = shards-1
    count = np.zeros((1,128),np.uint32)
    count[0,0] = valid
    grouped, counts, offsets = module.pallas_collector_partition(
        jnp.asarray(words),jnp.asarray(ids[None]),jnp.asarray(count),
        shard_count=shards,interpret=True)
    order = sorted(range(valid),key=lambda i:(int(ids[i]),i))
    expected = np.zeros_like(words)
    expected[6] = np.uint32(0xffffffff)
    expected[:,:valid] = words[:,order]
    np.testing.assert_array_equal(grouped,expected)
    totals = np.zeros((1,128),np.uint32)
    totals[0,:shards] = np.bincount(ids[:valid],minlength=shards)
    starts = np.zeros_like(totals)
    starts[0,1:shards+1] = np.cumsum(totals[0,:shards])
    np.testing.assert_array_equal(counts,totals)
    np.testing.assert_array_equal(offsets,starts)
