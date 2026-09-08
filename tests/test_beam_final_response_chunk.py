from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest
import jax


@pytest.mark.parametrize('failed',[False,True])
def test_response_chunk_transports_bytes_or_preserves_preparation_failure(failed):
    from tpu_beam_search.beam_final_response_chunk import make_final_response_chunk_call
    from tpu_beam_search.beam_final_response_routing import pallas_prepare_final_response_exchange
    wire=(np.arange(256*128,dtype=np.uint32).reshape(256,128)//7).astype(np.uint8)
    requests=np.zeros((4,256),np.uint32)
    control=np.zeros((2,128),np.uint32)
    control[0,0]=129
    validation=np.zeros((2,128),np.uint32)
    validation[0,0]=int(failed)
    grouped,intervals,error=pallas_prepare_final_response_exchange(
        *map(jnp.asarray,(wire,requests,control,validation)),world_size=1,interpret=True)
    call=make_final_response_chunk_call(SimpleNamespace(size=1),wire_width=128,interpret=True)
    for epoch,length in ((0,128),(1,1),(2,0)):
        received,status=map(np.asarray,call(grouped,intervals,error,jnp.array([epoch],jnp.uint32)))
        expected=np.zeros((128,128),np.uint8)
        if not failed:
            expected[:length]=wire[epoch*128:epoch*128+length]
        np.testing.assert_array_equal(received,expected)
        assert status[0,0]==(0 if failed else length)
        assert status[1,0]==int(failed)


def test_response_epoch_eight_rank_abi_traces_without_host_readback():
    from tpu_beam_search.beam_final_response_chunk import make_final_response_chunk_call
    call=make_final_response_chunk_call(SimpleNamespace(size=8),wire_width=128)
    trace=jax.make_jaxpr(call,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((35,1024),jnp.uint32),
        jax.ShapeDtypeStruct((3,128),jnp.uint32),
        jax.ShapeDtypeStruct((1,128),jnp.uint32),
        jax.ShapeDtypeStruct((1,),jnp.uint32))
    assert [(x.shape,x.dtype) for x in trace.out_avals]==[
        ((1024,128),jnp.dtype('uint8')),((2,128),jnp.dtype('uint32'))]
