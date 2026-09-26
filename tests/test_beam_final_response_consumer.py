from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest


def packet(targets,values,error=0):
    wire=np.zeros((128,256),np.uint8)
    for i,(target,value) in enumerate(zip(targets,values,strict=True)):
        wire[i,:150]=value
        wire[i,150:154]=np.frombuffer(int(target).to_bytes(4,'little'),dtype=np.uint8)
    status=np.zeros((2,128),np.uint32)
    status[0,0]=len(targets)
    status[1,0]=error
    return jnp.asarray(wire),jnp.asarray(status)


def test_response_consumer_preserves_private_frontier_after_cross_epoch_duplicate():
    from tpu_beam_search.beam_final_response_consumer import make_final_response_consumer
    call=make_final_response_consumer(SimpleNamespace(size=1),state_len=150,interpret=True)
    frontier=jnp.full((3,160),99,jnp.uint8)
    seen=jnp.zeros((1,128),jnp.uint32)
    control=jnp.zeros((2,128),jnp.uint32)
    common=jnp.zeros((1,128),jnp.uint32)
    count=jnp.array([3],jnp.uint32)
    frontier,seen,control,common=call(frontier,seen,control,*packet([2],[7]),count,common)
    expected=np.full((3,160),99,np.uint8)
    expected[2,:150]=7
    expected[2,150:]=0
    np.testing.assert_array_equal(frontier,expected)
    assert not np.asarray(common).any()
    frontier,seen,control,common=call(frontier,seen,control,*packet([0,2],[8,9]),count,common)
    np.testing.assert_array_equal(frontier,expected)  # whole rejected batch, including target0
    assert int(common[0,0])==1 and int(control[1,0])==1
    frontier,seen,control,common=call(frontier,seen,control,*packet([1],[10]),count,common)
    np.testing.assert_array_equal(frontier,expected)


@pytest.mark.parametrize('error',[1,0x80000000])
def test_empty_received_error_still_blocks_consumer(error):
    from tpu_beam_search.beam_final_response_consumer import make_final_response_consumer
    call=make_final_response_consumer(SimpleNamespace(size=1),state_len=150,interpret=True)
    frontier,seen,control,common=call(jnp.full((3,160),42,jnp.uint8),
        jnp.zeros((1,128),jnp.uint32),jnp.zeros((2,128),jnp.uint32),
        *packet([],[],error),jnp.array([3],jnp.uint32),jnp.zeros((1,128),jnp.uint32))
    assert np.all(np.asarray(frontier)==42) and not np.asarray(seen).any()
    assert int(control[1,0])==1 and int(common[0,0])==1


def test_eight_rank_consumer_traces_collective_path_without_host_counts():
    import jax
    from tpu_beam_search.beam_final_response_consumer import make_final_response_consumer
    call=make_final_response_consumer(SimpleNamespace(size=8),state_len=150)
    shapes=[((256,160),jnp.uint8),((1,256),jnp.uint32),((2,128),jnp.uint32),
            ((1024,256),jnp.uint8),((2,128),jnp.uint32),((1,),jnp.uint32),
            ((1,128),jnp.uint32)]
    trace=jax.make_jaxpr(call,axis_env=[('core',8)])(
        *(jax.ShapeDtypeStruct(*spec) for spec in shapes))
    assert [(v.shape,v.dtype) for v in trace.out_avals]==[
        ((256,160),jnp.dtype('uint8')),((1,256),jnp.dtype('uint32')),
        ((2,128),jnp.dtype('uint32')),((1,128),jnp.dtype('uint32'))]
