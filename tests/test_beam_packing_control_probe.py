from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('stage',['packing_control','packing_selection'])
@pytest.mark.parametrize('bad',[False,True])
def test_packing_probe_keeps_runtime_selection_and_error_observable(stage,bad):
    from benchmarks.beam_response_isolation_probe import stage_call
    call,_ = stage_call(stage,SimpleNamespace(size=8),interpret=True)
    intervals = np.zeros((3,128),np.uint32)
    intervals[0,:8] = np.arange(8)*129
    intervals[1,:8] = np.arange(8)+5
    prior = np.zeros((1,128),np.uint32)
    prior[0,0] = int(bad)
    payload,controls = map(np.asarray,call(jnp.zeros((32,2048),jnp.uint32),
        jnp.asarray(intervals),jnp.asarray(prior),jnp.array([3],jnp.uint32)))
    assert not payload.any()
    expected = np.zeros((8,2,128),np.uint32)
    expected[:,1,0] = int(bad)
    if stage == 'packing_selection':
        expected[:,0,0] = np.arange(8)*129
        expected[:,0,1] = np.arange(8)+5
        expected[:,0,2] = 384
    np.testing.assert_array_equal(controls,expected)
