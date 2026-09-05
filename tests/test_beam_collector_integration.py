import jax.numpy as jnp
import numpy as np

from tpu_beam_search import beam_collector as module
from test_beam_hash import oracle


def test_hash_collector_fills_siblings_then_rejects_whole_input():
    assert hasattr(module,'pallas_collect')
    words = np.random.default_rng(128).integers(0,2**32,(8,256),dtype=np.uint32)
    valid,shards = 213,3
    ids = []
    for i in range(valid):
        lo = int(words[0,i]) | (int(words[1,i]) << 32)
        hi = int(words[2,i]) | (int(words[3,i]) << 32)
        ids.append(oracle(lo,hi,0x13198a2e03707344)%shards)
    groups = [words[:,np.flatnonzero(np.asarray(ids) == s)] for s in range(shards)]
    assert all(64 < g.shape[1] <= 128 for g in groups)
    a = np.zeros((shards,8,128),np.uint32)
    b = np.zeros_like(a)
    c = np.zeros((shards,8,128),np.uint32)
    count = np.zeros((1,128),np.uint32)
    count[0,0] = valid
    ea,eb,ec = a.copy(),b.copy(),c.copy()
    for step in range(3):
        a,b,c,f = module.pallas_collect(
            *(jnp.asarray(x) for x in (a,b,words,c,count)),interpret=True)
        if step < 2:
            for s,g in enumerate(groups):
                (ea if step == 0 else eb)[s,:,:g.shape[1]] = g
                ec[s,2+step,0] = g.shape[1]
                ec[s,6,0] = step
        else:
            ec[:,7,0] = 1
        np.testing.assert_array_equal(a,ea)
        np.testing.assert_array_equal(b,eb)
        np.testing.assert_array_equal(c,ec)
        expected_f = np.zeros((1,128),np.uint32)
        expected_f[0,0] = int(step == 2)
        np.testing.assert_array_equal(f,expected_f)
