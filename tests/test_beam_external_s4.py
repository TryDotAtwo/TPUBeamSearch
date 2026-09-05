import os
import subprocess
import pytest
import jax.numpy as jnp
import numpy as np


def test_external_s4_uses_parent_high_low_route_not_payload_and_keeps_all_unique():
    from tpu_beam_search import beam_external_sort as module
    assert hasattr(module,'pallas_external_stream4_dedup')
    w = np.zeros((8,256),np.uint32)
    w[0] = np.arange(256,dtype=np.uint32)
    w[6] = 7
    w[0,128:132] = 1
    w[4,1],w[5,1] = 0,2
    w[4,128],w[5,128] = 100,1
    w[4,129],w[5,129],w[7,129] = 99,1,10
    w[4,130],w[5,130],w[7,130] = 99,1,9
    w[6,131] = 8
    keep = [0,130,*range(2,128),*range(132,256)]
    expected = np.zeros_like(w)
    expected[6] = np.uint32(0xffffffff)
    expected[:,:252] = w[:,keep]
    actual,count = module.pallas_external_stream4_dedup(jnp.asarray(w),
        jnp.array([256],jnp.uint32),jnp.array([7],jnp.uint32),interpret=True)
    np.testing.assert_array_equal(actual,expected)
    assert int(count[0,0]) == 252


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'),reason='source CPU oracle not configured')
@pytest.mark.parametrize('valid,threshold',[(0,0xffffffff),(251,5)])
def test_external_s4_matches_original_cpp_with_poisoned_tail(valid,threshold):
    from tpu_beam_search.beam_external_sort import pallas_external_stream4_dedup
    rng = np.random.default_rng(71421)
    w = rng.integers(0,2**32,(8,256),dtype=np.uint32)
    w[:4] = 0
    w[0] = rng.integers(0,35,256,dtype=np.uint32)
    w[3] = np.uint32(0x80000000)
    w[6] = rng.integers(0,8,256,dtype=np.uint32)
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE'],'dedup'],
        input=' '.join(map(str,[valid,threshold,*w[:,:valid].T.ravel()])),
        capture_output=True,text=True,check=True,timeout=30)
    records = np.asarray([list(map(int,line.split())) for line in raw.stdout.splitlines()],
                         np.uint32).reshape(-1,8)
    want = np.zeros_like(w)
    want[6] = np.uint32(0xffffffff)
    want[:,:len(records)] = records.T
    actual,count = pallas_external_stream4_dedup(jnp.asarray(w),jnp.array([valid],jnp.uint32),
        jnp.array([threshold],jnp.uint32),interpret=True)
    np.testing.assert_array_equal(actual,want)
    assert int(count[0,0]) == len(records)
