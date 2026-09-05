import os
import subprocess
import numpy as np
import jax.numpy as jnp
import pytest
from tpu_beam_search import beam_external_sort as module


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'), reason='original C++ oracle not configured')
def test_composed_external_stream3_matches_original_cpp_winners_and_routing():
    assert hasattr(module, 'pallas_external_stream3')
    n, rank, world, threshold = 256, 3, 8, 5
    rng = np.random.default_rng(6581)
    words = np.zeros((8,n), np.uint32)
    words[0] = rng.integers(0,70,n,dtype=np.uint32)
    words[3] = np.uint32(0x80000000)
    words[4] = np.arange(n,dtype=np.uint32)
    words[5] = np.uint32(0x10000)
    words[6] = rng.integers(0,8,n,dtype=np.uint32)
    words[7] = np.arange(n,dtype=np.uint32) % 24
    payload = np.arange(n-1,-1,-1,dtype=np.uint32)[None]
    tokens = [n,threshold,rank,world,*np.concatenate((words,payload)).T.ravel()]
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE'],'stream3'],
        input=' '.join(map(str,tokens)),text=True,capture_output=True,check=True,timeout=30)
    lines = raw.stdout.splitlines()
    local_n, remote_n = map(int,lines[1].split())
    expected = np.array([list(map(int,row.split())) for row in lines[4:]],np.uint32).reshape(-1,8)
    lo, ro, lc, counts, offsets = module.pallas_external_stream3(
        jnp.asarray(words),jnp.asarray(payload),jnp.array([n],jnp.uint32),
        jnp.array([threshold],jnp.uint32),local_rank=rank,world_size=world,interpret=True)
    assert int(lc[0,0]) == local_n
    np.testing.assert_array_equal(counts[0,:world],list(map(int,lines[2].split())))
    np.testing.assert_array_equal(offsets[0,:world+1],list(map(int,lines[3].split())))
    np.testing.assert_array_equal(lo[:,:local_n].T,expected[:local_n])
    np.testing.assert_array_equal(ro[:,:remote_n].T,expected[local_n:])
    for result, valid in ((lo,local_n),(ro,remote_n)):
        neutral = np.zeros((8,n-valid),np.uint32)
        neutral[6] = np.uint32(0xffffffff)
        np.testing.assert_array_equal(result[:,valid:],neutral)
