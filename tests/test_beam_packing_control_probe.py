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


@pytest.mark.parametrize('bad',[False,True])
def test_guard_probe_exposes_only_live_interval_geometry(bad):
    # Removing the guard would expose geometry for empty/exhausted peers.
    from benchmarks.beam_packing_control_probe import make_probe
    call=make_probe(selection=True,guard=True,world_size=8,interpret=True)
    intervals=np.zeros((3,128),np.uint32)
    intervals[0,:8]=[1,127,128,129,255,256,1000,2048]
    intervals[1,:8]=[129,257,128,0,200,129,300,0]
    prior=np.zeros((1,128),np.uint32)
    prior[0,0]=int(bad)
    payload,controls=map(np.asarray,call(jnp.zeros((32,2048),jnp.uint32),
        jnp.asarray(intervals),jnp.asarray(prior),jnp.array([1],jnp.uint32)))
    expected=np.zeros((8,2,128),np.uint32)
    expected[:,1,0]=int(bad)
    if not bad:
        for peer in range(8):
            start,count=map(int,intervals[:2,peer])
            if count>128:
                begin=start+128
                expected[peer,0,:4]=[min(128,count-128),begin,begin//128*128,begin%128]
    assert not payload.any()
    np.testing.assert_array_equal(controls,expected)


@pytest.mark.parametrize('start,count,error',[
    (2048,0,0), (2048,1,1), (0,2049,1),
    (0xffffffff,1,1), (0,0xffffffff,1),
])
def test_selection_probe_preserves_invalid_unsigned_values(start,count,error):
    from benchmarks.beam_response_isolation_probe import stage_call
    call,_=stage_call('packing_selection',SimpleNamespace(size=8),interpret=True)
    intervals=np.zeros((3,128),np.uint32)
    intervals[0,7]=start
    intervals[1,7]=count
    _,controls=map(np.asarray,call(jnp.zeros((32,2048),jnp.uint32),
        jnp.asarray(intervals),jnp.zeros((1,128),jnp.uint32),
        jnp.array([0xffffffff],jnp.uint32)))
    expected=np.zeros((8,2,128),np.uint32)
    expected[:,0,2]=2048  # saturated offset, not wrapped UINT_MAX*128
    expected[7,0,:2]=[start,count]
    expected[:,1,0]=error  # any bad live peer poisons every local output peer
    np.testing.assert_array_equal(controls,expected)
