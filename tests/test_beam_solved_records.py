import numpy as np
import jax.numpy as jnp
import jax
import pytest


@pytest.mark.parametrize('enable_x64',[False,True])
def test_solved_records_preserve_parent_carry_route_depth_suffix(enable_x64):
    from tpu_beam_search.beam_solved_records import pallas_solved_records
    hashes = np.arange(1024,dtype=np.uint32).reshape(4,256)
    suffix = np.arange(256,dtype=np.uint32)[None,:]
    base = np.array([0xfffffffe,7],np.uint32)
    depth = np.array([19],np.uint32)
    with jax.enable_x64(enable_x64):
        got = pallas_solved_records(*map(jnp.asarray,(hashes,suffix,base,depth)),
            move_count=24,local_rank=7,interpret=True)
    want = np.zeros((10,256),np.uint32)
    want[:4] = hashes
    for index in range(256):
        parent = (7<<32)+0xfffffffe+index//24
        want[4,index],want[5,index] = parent&0xffffffff,parent>>32
        want[7,index] = (7<<16)|(7<<8)|(index%24)
    want[8] = 19
    want[9] = suffix[0]
    np.testing.assert_array_equal(got,want)
