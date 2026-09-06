import numpy as np
import jax.numpy as jnp


def test_rank_prefixes_preserve_carry_and_ignore_padding():
    from tpu_beam_search.beam_final_prefix import pallas_final_prefixes
    counts = np.full((2,128),0xffffffff,np.uint32)
    counts[:,:8] = [[0xffffffff,2,0,9,3,0,5,1],[7,0,0xffffffff,2,0,11,0,1]]
    bases,totals = pallas_final_prefixes(jnp.asarray(counts),world_size=8,interpret=True)
    less,equal = [list(map(int,row[:8])) for row in counts]
    expected = np.zeros((4,128),np.uint32)
    for rank in range(8):
        for phase,value in enumerate((sum(less[:rank]),sum(less)+sum(equal[:rank]))):
            expected[2*phase,rank],expected[2*phase+1,rank] = value&0xffffffff,value>>32
    np.testing.assert_array_equal(bases,expected)
    total = np.zeros((4,128),np.uint32)
    for phase,value in enumerate((sum(less),sum(equal))):
        total[2*phase,0],total[2*phase+1,0] = value&0xffffffff,value>>32
    np.testing.assert_array_equal(totals,total)
