import numpy as np
import jax
import jax.numpy as jnp


def test_final_plan_separates_source_owner_return_and_preserves_parent64():
    from tpu_beam_search.beam_final_plan import pallas_final_plan
    count, world = 19, 8
    meta = np.zeros((8,128),np.uint32)
    meta[4] = np.arange(128,dtype=np.uint32)
    meta[5] = 0xabcdef01
    meta[7] = (6 << 16) | (4 << 8) | 23
    indices = np.zeros((2,128),np.uint32)
    indices[0] = np.arange(128,dtype=np.uint32)
    bounds = np.zeros((2,128),np.uint32)
    bounds[0,:world+1] = [(r*count+world-1)//world for r in range(world+1)]
    fn = jax.jit(lambda m,i,b: pallas_final_plan(m,i,b,world_size=world,interpret=True))
    requests,sources,valid = fn(*map(jnp.asarray,(meta,indices,bounds)))
    np.testing.assert_array_equal(sources,6)
    np.testing.assert_array_equal(requests[:2],meta[4:6])
    np.testing.assert_array_equal(valid[0],np.arange(128)<count)
    for i in range(count):
        rank = i*world//count
        assert int(requests[2,i]) == i-int(bounds[0,rank])
        assert int(requests[3,i]) == rank | (23<<16)
