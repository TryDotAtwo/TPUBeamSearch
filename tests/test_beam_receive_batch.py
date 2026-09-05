import importlib.util
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('empty',[False,True])
def test_receive_batch_compacts_counts_and_restores_source_rank_order(empty):
    assert importlib.util.find_spec('tpu_beam_search.beam_receive_batch') is not None
    from tpu_beam_search.beam_receive_batch import pallas_compact_received
    snapshots = np.arange(7*8*128,dtype=np.uint32).reshape(7,8,128)+17
    counts = np.zeros((7,128),np.uint32)
    counts[:,0] = 0 if empty else [0,127,1,128,17,0,2]
    rank = np.zeros((1,128),np.uint32)
    rank[0,0] = 3
    packed,total = pallas_compact_received(
        *(jnp.asarray(x) for x in (snapshots,counts,rank)),interpret=True)
    order = sorted(range(7),key=lambda e:(3-e-1)%8)
    expected = np.zeros((8,1024),np.uint32)
    expected[6] = np.uint32(0xffffffff)
    cursor = 0
    for e in order:
        n = int(counts[e,0])
        expected[:,cursor:cursor+n] = snapshots[e,:,:n]
        cursor += n
    wanted_count = np.zeros((1,128),np.uint32)
    wanted_count[0,0] = cursor
    np.testing.assert_array_equal(packed,expected)
    np.testing.assert_array_equal(total,wanted_count)
