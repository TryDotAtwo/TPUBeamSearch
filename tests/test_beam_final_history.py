import numpy as np
import jax.numpy as jnp


def test_final_history_records_preserve_source_and_mask_padding():
    from tpu_beam_search.beam_final_history import pallas_final_history_records
    meta=np.zeros((8,128),np.uint32)
    meta[4,:]=9
    meta[5,:]=1
    meta[7,:]=(6<<16)|(4<<8)|23
    target=np.arange(128,dtype=np.uint32)[None,:]
    valid=np.zeros((1,128),np.uint32)
    valid[0,:3]=1
    got=np.asarray(pallas_final_history_records(jnp.asarray(meta),
        jnp.asarray(target),jnp.asarray(valid),interpret=True))
    expected=np.zeros((5,128),np.uint32)
    expected[:3,:3]=meta[[4,5,7],:3]
    expected[3,:3]=target[0,:3]
    expected[4,:3]=1
    np.testing.assert_array_equal(got,expected)
