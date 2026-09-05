import numpy as np
import jax.numpy as jnp


def test_first_suffix_hit_preserves_immediate_hash_and_earliest_solution():
    from tpu_beam_search.beam_suffix_hit import pallas_merge_suffix_hit
    immediate = np.arange(512,dtype=np.uint32).reshape(4,128)
    found = immediate.copy()
    flags = np.zeros((1,128),np.uint32)
    flags[0,0] = 1  # Immediate/K1 solution must win.
    ids = np.zeros((1,128),np.uint32)
    valid = np.zeros((1,128),np.uint32)
    valid[0,:4] = 1
    for suffix in (1,2,3):
        projected = immediate + 1000*suffix
        hit = np.ones((1,128),np.uint32)
        if suffix == 1:
            hit[0,2] = 0
        result = pallas_merge_suffix_hit(*map(jnp.asarray,
            (found,flags,ids,projected,hit,valid)),suffix_id=suffix,interpret=True)
        found,flags,ids = map(np.asarray,result)
    np.testing.assert_array_equal(ids[0,:4],[0,1,2,1])
    np.testing.assert_array_equal(found[:,0],immediate[:,0])
    np.testing.assert_array_equal(found[:,1],immediate[:,1]+1000)
    np.testing.assert_array_equal(found[:,2],immediate[:,2]+2000)
    np.testing.assert_array_equal(found[:,4:],immediate[:,4:])
    np.testing.assert_array_equal(flags[0,:4],1)
    np.testing.assert_array_equal(flags[0,4:],0)
