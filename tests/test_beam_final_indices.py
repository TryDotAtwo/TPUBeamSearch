import numpy as np
import jax.numpy as jnp


def test_indices_carry_cap_sentinel_and_error_gate():
    from tpu_beam_search.beam_final_indices import pallas_final_indices
    ordinal = np.full((2,1,128),0xffffffff,np.uint32)
    ordinal[0,0,:4] = [0,1,2,3]
    ordinal[1,0,:4] = [0,1,2,3]
    bases = np.zeros((4,128),np.uint32)
    bases[:,3] = [0xfffffffe,0,1,1]
    keep = np.zeros((2,128),np.uint32)
    keep[:,0] = [3,1]
    for bad in (0,1):
        error = np.zeros((1,128),np.uint32)
        error[0,0] = bad
        index,valid = pallas_final_indices(*map(jnp.asarray,(ordinal,bases,keep,error)),rank=3,interpret=True)
        want = np.zeros((2,2,1,128),np.uint32)
        mask = np.zeros_like(ordinal)
        for phase,base in enumerate((0xfffffffe,0x100000001)):
            for lane in range(4):
                value = base+lane
                if not bad and value<0x100000003:
                    want[phase,:,0,lane] = value&0xffffffff,value>>32
                    mask[phase,0,lane] = 1
        np.testing.assert_array_equal(index,want)
        np.testing.assert_array_equal(valid,mask)
