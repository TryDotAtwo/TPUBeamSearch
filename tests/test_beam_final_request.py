import numpy as np
import jax.numpy as jnp


def test_final_request_preserves_parent_and_separates_source_from_return_rank():
    from tpu_beam_search.beam_final_request import pallas_final_requests
    meta = np.zeros((8,256),np.uint32)
    meta[4] = np.arange(256,dtype=np.uint32)+0xffffff00
    meta[5] = 0x87654321
    meta[7] = (7<<16)|(4<<8)|23
    targets = np.arange(256,dtype=np.uint32)[None]+0x80000000
    ranks = np.full((1,256),3,np.uint32)
    requests,sources = pallas_final_requests(*map(jnp.asarray,(meta,targets,ranks)),interpret=True)
    expected = np.stack((meta[4],meta[5],targets[0],ranks[0]|np.uint32(23<<16)))
    np.testing.assert_array_equal(requests,expected)
    np.testing.assert_array_equal(sources,np.full((1,256),7,np.uint32))
