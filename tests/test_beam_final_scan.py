import numpy as np
import jax.numpy as jnp


def test_phase_ordinals_cross_tiles_shards_and_empty_gaps():
    from tpu_beam_search.beam_final_scan import pallas_final_phase_scan
    masks = np.zeros((2,3,256),np.uint32)
    masks[0,0,[0,127,128,255]] = 1
    masks[0,2,[0,129]] = 1
    masks[1,1,:] = 1
    masks[1,2,[0,255]] = 1
    for values in (masks,np.zeros_like(masks),np.ones_like(masks)):
        ordinal,count = pallas_final_phase_scan(jnp.asarray(values),interpret=True)
        expected = np.full_like(values,0xffffffff)
        for phase in range(2):
            selected = values[phase].astype(bool)
            expected[phase][selected] = np.arange(np.count_nonzero(selected),dtype=np.uint32)
        np.testing.assert_array_equal(ordinal,expected)
        expected_count = np.zeros((2,128),np.uint32)
        expected_count[:,0] = values.sum(axis=(1,2))
        np.testing.assert_array_equal(count,expected_count)
