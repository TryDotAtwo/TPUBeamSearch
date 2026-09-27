from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('prior_error',[False,True])
def test_single_rank_epoch_loop_materializes_and_checks_both_streams(prior_error):
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState
    from tpu_beam_search.beam_final_materialization_epochs import make_final_materialization_epochs
    from tpu_beam_search.beam_final_delivery import pallas_final_delivery_plan
    from tpu_beam_search.beam_final_delivery_exchange import pallas_prepare_delivery_exchange

    mesh = SimpleNamespace(size=1)
    zero = jnp.zeros((1,128),jnp.uint32)
    state = FinalMaterializationState(jnp.zeros((3,160),jnp.uint8),
        jnp.zeros((1,5,128),jnp.uint32),zero,jnp.zeros((2,128),jnp.uint32),
        zero,jnp.zeros((2,128),jnp.uint32),
        zero.at[0,0].set(1) if prior_error else zero)
    packed = np.zeros((11,128),np.uint32)
    packed[4,0] = 1
    packed[7,0] = 0
    packed[8,0] = 0
    packed[10,0] = 1
    plan = pallas_final_delivery_plan(jnp.asarray(packed),
        jnp.zeros((2,128),jnp.uint32).at[0,0].set(1),zero,
        world_size=1,interpret=True)
    prepared = pallas_prepare_delivery_exchange(plan,world_size=1,interpret=True)
    parents = np.zeros((2,256),np.uint8)
    parents[1,:3] = [4,5,6]
    generators = np.tile(np.arange(256,dtype=np.int32),(2,1))
    generators[0,:3] = [1,2,0]
    call = make_final_materialization_epochs(mesh,state_len=150,move_count=2,
        request_capacity=prepared[0].shape[1],history_capacity=prepared[2].shape[1],
        interpret=True)
    result,response_error,history_error = call(state,jnp.asarray(parents),
        jnp.asarray(generators),prepared,plan.target_counts[0,:1],
        jnp.array([1],jnp.uint32))
    if prior_error:
        assert int(result.error[0,0])==1
        assert not np.asarray(result.frontier).any()
        assert not np.asarray(result.history).any()
        return
    assert not np.asarray(result.error).any()
    assert not np.asarray(response_error).any()
    assert not np.asarray(history_error).any()
    np.testing.assert_array_equal(np.asarray(result.frontier)[0,:3],[5,6,4])
    np.testing.assert_array_equal(np.asarray(result.history)[0,:,0],[1,0,0,0,1])


def test_eight_rank_epoch_loop_traces_dynamic_common_bound():
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState
    from tpu_beam_search.beam_final_materialization_epochs import make_final_materialization_epochs
    u32=lambda shape:jax.ShapeDtypeStruct(shape,jnp.uint32)
    state=FinalMaterializationState(jax.ShapeDtypeStruct((128,160),jnp.uint8),
        u32((1,5,128)),u32((1,128)),u32((2,128)),u32((1,128)),
        u32((2,128)),u32((1,128)))
    prepared=(u32((7,128)),u32((3,128)),u32((8,128)),u32((3,128)),u32((1,128)))
    call=make_final_materialization_epochs(SimpleNamespace(size=8),
        state_len=150,move_count=2,request_capacity=128,history_capacity=128)
    trace=jax.make_jaxpr(call,axis_env=[('core',8)])(state,
        jax.ShapeDtypeStruct((128,256),jnp.uint8),
        jax.ShapeDtypeStruct((2,256),jnp.int32),prepared,u32((8,)),u32((1,)))
    assert [value.shape for value in trace.out_avals]==[
        (128,160),(1,5,128),(1,128),(2,128),(1,128),(2,128),(1,128),
        (1,128),(1,128)]
