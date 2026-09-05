import os
import subprocess
import importlib.util
import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.skipif(not os.environ.get('BEAM_SOURCE_ORACLE'),reason='source oracle not configured')
@pytest.mark.parametrize('buckets',[1,32,1024])
def test_pallas_k1_fingerprint_and_buckets_match_original_cpp(buckets):
    assert importlib.util.find_spec('tpu_beam_search.beam_k1_keys') is not None
    from tpu_beam_search.beam_k1_keys import pallas_k1_keys
    words = np.random.default_rng(605).integers(0,2**32,(4,256),dtype=np.uint32)
    words[:,0],words[:,1] = 0,0xffffffff
    # hi=0, lo=salt XOR mix(golden_ratio): distribution input and output are0.
    words[:,2] = [1384316031,1177128987,0,0]
    raw = subprocess.run([os.environ['BEAM_SOURCE_ORACLE'],'k1keys'],
        input=' '.join(map(str,[256,buckets,*words.T.ravel()])),
        text=True,capture_output=True,timeout=30)
    assert raw.returncode == 0, raw.stderr
    expected = np.array([list(map(int,line.split())) for line in raw.stdout.splitlines()],np.uint32).T
    actual = pallas_k1_keys(jnp.asarray(words),bucket_count=buckets,interpret=True)
    np.testing.assert_array_equal(actual,expected)
    assert np.all(expected[0] != 0)
    assert expected[0,2] == 1
