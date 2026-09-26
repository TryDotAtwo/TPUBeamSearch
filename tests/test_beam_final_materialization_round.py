from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('late_history_error',[False,True])
def test_final_round_delivers_children_and_matching_history_then_checks_coverage(late_history_error):
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState,make_final_materialization_round
    from tpu_beam_search.beam_final_delivery import pallas_final_delivery_plan
    from tpu_beam_search.beam_final_delivery_exchange import pallas_prepare_delivery_exchange
    from tpu_beam_search.beam_final_streaming_agreement import make_final_streaming_agreement
    mesh=SimpleNamespace(size=1)
    zero=jnp.zeros((1,128),jnp.uint32)
    state=FinalMaterializationState(jnp.full((3,160),99,jnp.uint8),
        jnp.zeros((1,5,128),jnp.uint32),zero,jnp.zeros((2,128),jnp.uint32),
        zero,jnp.zeros((2,128),jnp.uint32),zero)
    packed=np.zeros((11,128),np.uint32)
    packed[4,:2]=[1,0]
    packed[7,:2]=[1,0]
    packed[8,:2]=[0,1]
    packed[10,:2]=1
    plan=pallas_final_delivery_plan(jnp.asarray(packed),
        jnp.zeros((2,128),jnp.uint32).at[0,0].set(2),zero,world_size=1,interpret=True)
    prepared=pallas_prepare_delivery_exchange(plan,world_size=1,interpret=True)
    if late_history_error:
        prepared=(*prepared[:3],prepared[3].at[2,0].set(1),prepared[4])
    parents=np.zeros((2,256),np.uint8)
    parents[:,:3]=[[1,2,3],[4,5,6]]
    generators=np.tile(np.arange(256,dtype=np.int32),(2,1))
    generators[1,:3]=[1,2,0]
    call=make_final_materialization_round(mesh,state_len=150,move_count=2,interpret=True)
    state=call(state,jnp.asarray(parents),jnp.asarray(generators),prepared,
        plan.target_counts[0,:mesh.size],jnp.array([0],jnp.uint32))
    if late_history_error:
        assert np.all(np.asarray(state.frontier)==99)
        assert not np.asarray(state.history).any()
        assert not np.asarray(state.response_marks).any()
        assert int(state.error[0,0])==1
        return
    expected=np.full((3,160),99,np.uint8)
    expected[:2]=0
    expected[0,:3]=[5,6,4]
    expected[1,:3]=[1,2,3]
    np.testing.assert_array_equal(state.frontier,expected)
    np.testing.assert_array_equal(np.asarray(state.history)[0,:,:2],[[1,0],[0,0],[1,0],[0,1],[1,1]])
    common,_,_=make_final_streaming_agreement(mesh,interpret=True)(state.response_marks,
        state.response_control,state.history_marks,state.history_control,jnp.array([2],jnp.uint32),state.error)
    assert not np.asarray(common).any()


def test_eight_rank_materialization_round_traces_uniform_subepochs():
    import jax
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState,make_final_materialization_round
    shape=lambda s:jax.ShapeDtypeStruct(s,jnp.uint32)
    state=FinalMaterializationState(jax.ShapeDtypeStruct((128,160),jnp.uint8),
        shape((1,5,128)),shape((1,128)),shape((2,128)),shape((1,128)),
        shape((2,128)),shape((1,128)))
    prepared=(shape((7,128)),shape((3,128)),shape((8,128)),shape((3,128)),shape((1,128)))
    call=make_final_materialization_round(SimpleNamespace(size=8),state_len=150,move_count=2)
    trace=jax.make_jaxpr(call,axis_env=[('core',8)])(state,
        jax.ShapeDtypeStruct((128,256),jnp.uint8),jax.ShapeDtypeStruct((2,256),jnp.int32),
        prepared,shape((8,)),shape((1,)))
    assert [v.shape for v in trace.out_avals]==[
        (128,160),(1,5,128),(1,128),(2,128),(1,128),(2,128),(1,128)]
