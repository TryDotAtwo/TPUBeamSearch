import numpy as np
import jax.numpy as jnp


def test_final_group_sorts_routes_stably_without_promoting_padding():
    from tpu_beam_search.beam_final_group import pallas_group_final_records
    payload=np.arange(4*128,dtype=np.uint32).reshape(4,128)
    ranks=np.zeros((1,128),np.uint32)
    ranks[0,:5]=[2,0,2,1,0]
    valid=np.zeros((1,128),np.uint32)
    valid[0,:5]=1
    got=np.asarray(pallas_group_final_records(*map(jnp.asarray,(payload,ranks,valid)),interpret=True))
    order=[1,4,3,0,2]
    np.testing.assert_array_equal(got[:4,:5],payload[:,order])
    np.testing.assert_array_equal(got[4,:5],[0,0,1,2,2])
    np.testing.assert_array_equal(got[5,:5],order)
    np.testing.assert_array_equal(got[6],np.arange(128)<5)


def test_final_group_cross_tile_sparse_and_empty():
    from tpu_beam_search.beam_final_group import pallas_group_final_records
    payload=np.arange(5*256,dtype=np.uint32).reshape(5,256)
    ranks=(np.arange(256,dtype=np.uint32)%8)[None,:]
    for slots in ([0,127,128,129,255],[]):
        valid=np.zeros((1,256),np.uint32)
        valid[0,slots]=1
        got=np.asarray(pallas_group_final_records(*map(jnp.asarray,(payload,ranks,valid)),interpret=True))
        order=sorted(slots,key=lambda i:(int(ranks[0,i]),i))
        np.testing.assert_array_equal(got[:5,:len(order)],payload[:,order])
        np.testing.assert_array_equal(got[6,:len(order)],order)
        np.testing.assert_array_equal(got[7],np.arange(256)<len(order))
