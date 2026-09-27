from types import SimpleNamespace

import jax
import jax.numpy as jnp


def test_eight_rank_final_depth_traces_selection_and_both_streams():
    from tpu_beam_search.beam_final_depth import make_final_depth_call
    from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState
    u32=lambda shape:jax.ShapeDtypeStruct(shape,jnp.uint32)
    state=FinalMaterializationState(jax.ShapeDtypeStruct((128,160),jnp.uint8),
        u32((1,5,128)),u32((1,128)),u32((2,128)),u32((1,128)),
        u32((2,128)),u32((1,128)))
    call=make_final_depth_call(SimpleNamespace(size=8),state_len=150,move_count=30)
    trace=jax.make_jaxpr(call,axis_env=[('core',8)])(
        u32((1,8,128)),u32((1,8,128)),u32((1,8,128)),
        u32((1,)),u32((2,128)),u32((1,128)),
        jax.ShapeDtypeStruct((128,256),jnp.uint8),
        jax.ShapeDtypeStruct((30,256),jnp.int32),state)
    assert [value.shape for value in trace.out_avals]==[
        (128,160),(1,5,128),(1,128),(2,128),(1,128),(2,128),(1,128),
        (8,),(2,128),(2,128),(1,128),(1,128)]
