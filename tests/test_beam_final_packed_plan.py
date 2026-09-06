import numpy as np
import jax.numpy as jnp


def test_packed_plan_does_not_reactivate_zero_padding():
    from tpu_beam_search.beam_final_plan import pallas_final_packed_plan
    packed=np.zeros((11,128),np.uint32)
    packed[4,:3]=[5,7,9]
    packed[5,:3]=17
    packed[7,:3]=(6<<16)|(4<<8)|23
    packed[8,:3]=[0,4,8]
    packed[10,:3]=1
    bounds=np.zeros((2,128),np.uint32)
    bounds[0,:9]=[(r*9+7)//8 for r in range(9)]
    request,source,valid=pallas_final_packed_plan(jnp.asarray(packed),jnp.asarray(bounds),world_size=8,interpret=True)
    np.testing.assert_array_equal(valid[0],np.arange(128)<3)
    np.testing.assert_array_equal(request[:2,:3],packed[4:6,:3])
    np.testing.assert_array_equal(source[0,:3],6)
    for slot,index in enumerate((0,4,8)):
        rank=index*8//9
        assert int(request[2,slot])==index-int(bounds[0,rank])
        assert int(request[3,slot])==rank|(23<<16)
