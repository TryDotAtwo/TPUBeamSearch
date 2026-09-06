import numpy as np
import jax.numpy as jnp


def test_final_compaction_keeps_phase_order_metadata_and_indices():
    from tpu_beam_search.beam_final_compact import pallas_final_compact
    meta = np.arange(8*128,dtype=np.uint32).reshape(8,1,128)
    indices = np.zeros((2,2,1,128),np.uint32)
    valid = np.zeros((2,1,128),np.uint32)
    for phase,slot,index in ((0,9,0xfffffffe),(0,127,0xffffffff),(1,2,0x100000001)):
        valid[phase,0,slot]=1
        indices[phase,:,0,slot]=index&0xffffffff,index>>32
    packed = np.asarray(pallas_final_compact(*map(jnp.asarray,(meta,indices,valid)),interpret=True))
    np.testing.assert_array_equal(packed[:8,:3],meta[:,0,[9,127,2]])
    np.testing.assert_array_equal(packed[8:10,:3],[[0xfffffffe,0xffffffff,1],[0,0,1]])
    np.testing.assert_array_equal(packed[10,:3],1)
    np.testing.assert_array_equal(packed[:,3:],0)


def test_three_shards_padding_and_empty_compaction():
    from tpu_beam_search.beam_final_compact import pallas_final_compact
    meta=np.arange(8*3*128,dtype=np.uint32).reshape(8,3,128)
    indices=np.zeros((2,2,3,128),np.uint32)
    valid=np.zeros((2,3,128),np.uint32)
    picks=[(0,2,127,5),(0,0,0,1),(1,1,128-1,9),(1,2,1,10)]
    for phase,shard,slot,index in picks:
        valid[phase,shard,slot]=1
        indices[phase,0,shard,slot]=index
    for empty in (False,True):
        mask=np.zeros_like(valid) if empty else valid
        packed=np.asarray(pallas_final_compact(*map(jnp.asarray,(meta,indices,mask)),interpret=True))
        assert packed.shape==(11,1024)
        expected=np.zeros_like(packed)
        if not empty:
            for out,(phase,shard,slot,index) in enumerate(sorted(picks,key=lambda x:x[3])):
                expected[:8,out]=meta[:,shard,slot]
                expected[8,out]=index
                expected[10,out]=1
        np.testing.assert_array_equal(packed,expected)
