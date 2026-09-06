from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def test_single_rank_exchange_and_collective_error_gate():
    from tpu_beam_search.beam_final_exchange import make_final_chunk_exchange
    fn=make_final_chunk_exchange(SimpleNamespace(size=1),planes=4,
        interpret=pltpu.InterpretParams(detect_races=True))
    payload=np.zeros((1,4,128),np.uint32)
    payload[0,:,:3]=np.arange(12,dtype=np.uint32).reshape(4,3)+1
    control=np.zeros((1,2,128),np.uint32)
    control[0,0,0]=3
    wire,counts,error=map(np.asarray,fn(jnp.asarray(payload),jnp.asarray(control)))
    np.testing.assert_array_equal(wire,payload)
    assert counts[0,0,0]==3
    assert not error.any()
    control[0,0,0]=0
    wire,counts,error=map(np.asarray,fn(jnp.asarray(payload),jnp.asarray(control)))
    assert not wire.any()
    assert not counts.any()
    assert not error.any()
    for bad_count,bad_flag in ((129,0),(3,1)):
        control[0,0,0],control[0,1,0]=bad_count,bad_flag
        wire,counts,error=map(np.asarray,fn(jnp.asarray(payload),jnp.asarray(control)))
        assert not wire.any()
        assert not counts.any()
        assert error[0,0]==1


def test_eight_rank_final_exchange_traces_typed_hbm_outputs():
    from tpu_beam_search.beam_final_exchange import make_final_chunk_exchange
    fn=make_final_chunk_exchange(SimpleNamespace(size=8),planes=4)
    traced=jax.make_jaxpr(fn,axis_env=[('core',8)])(
        jax.ShapeDtypeStruct((8,4,128),jnp.uint32),jax.ShapeDtypeStruct((8,2,128),jnp.uint32))
    assert tuple(x.shape for x in traced.out_avals)==((8,4,128),(8,1,128),(1,128))
    call=[e for e in traced.jaxpr.eqns if e.primitive.name=='pallas_call'][-1]
    assert tuple(a.shape for a in call.params['out_avals'])==((8,4,128),(8,1,128))
    assert all(a.memory_space==pltpu.HBM for a in call.params['out_avals'])
