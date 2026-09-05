import numpy as np
import jax.numpy as jnp


def test_phase_masks_exclude_dirty_padding_and_preserve_high_score_keys():
    from tpu_beam_search.beam_final_phase import pallas_final_phase_masks
    scores = np.full((2,256),0x80000001,np.uint32)
    scores[0,:4] = [0x80000000,0x80000001,0xffffffff,0]
    scores[1,127:130] = [0x80000000,0x80000001,0xffffffff]
    clean = np.zeros((1,128),np.uint32)
    clean[0,:2] = [3,130]
    for threshold in (0,0x80000001,0xffffffff):
        masks = pallas_final_phase_masks(jnp.asarray(scores),jnp.asarray(clean),
            jnp.array([threshold],jnp.uint32),interpret=True)
        valid = np.arange(256)[None,:] < clean[0,:2,None]
        expected = np.stack((valid & (scores<threshold),valid & (scores==threshold))).astype(np.uint32)
        np.testing.assert_array_equal(masks,expected)
