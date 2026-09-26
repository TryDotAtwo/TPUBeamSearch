from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def inputs(full=False):
    from tpu_beam_search.beam_runner import StreamRoundState
    a = jnp.zeros((1,8,128),jnp.uint32).at[:,6].set(0xffffffff)
    controls = jnp.zeros((1,8,128),jnp.uint32)
    if full: controls = controls.at[0,:2,0].set(128)
    hist = jnp.zeros((2,128),jnp.uint32)
    state = StreamRoundState(a,a,controls,hist,hist,jnp.zeros((1,128),jnp.uint32),
        jnp.zeros((4,128),jnp.uint32),jnp.zeros((2,128),jnp.uint32),
        jnp.zeros((2,128),jnp.uint32),jnp.zeros((1,128),jnp.uint32))
    words = jnp.zeros((8,128),jnp.uint32).at[6].set(0xffffffff)
    words = words.at[0,:3].set(jnp.array([7,7,9],jnp.uint32))
    words = words.at[4,:3].set(jnp.array([8,3,5],jnp.uint32))
    words = words.at[6,:3].set(jnp.array([2,1,2],jnp.uint32))
    return (state,words,jnp.arange(128,dtype=jnp.uint32)[None],jnp.array([3],jnp.uint32),
        jnp.zeros((2,128),jnp.uint32).at[0,0].set(1),jnp.zeros((1,),jnp.uint32),a[0])


@pytest.mark.parametrize('full',[False,True])
def test_actual_local_s3_collector_s4_s5_round(full):
    from tpu_beam_search.beam_runner import make_stream_round_call
    args = inputs(full)
    fn = make_stream_round_call(SimpleNamespace(size=1),bins=8,period=1,
        clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    state,error,jobs = fn(*args)
    assert int(error[0,0]) == int(full)
    if full:
        np.testing.assert_array_equal(state.a,args[0].a)
        np.testing.assert_array_equal(state.b,args[0].b)
        assert int(state.controls[0,7,0]) == 1
        assert not np.asarray(jobs).any()
        np.testing.assert_array_equal(state.epoch,args[0].epoch)
    else:
        np.testing.assert_array_equal(np.asarray(state.a)[0,[0,4,6],:2],[[7,9],[3,5],[1,2]])
        assert int(state.controls[0,0,0]) == 2
        assert not np.asarray(state.controls[0,2:6]).any()
        np.testing.assert_array_equal(state.threshold_b[:,0],[1,1])
        np.testing.assert_array_equal(state.epoch[:,0],[0,1,0,0])
        assert int(jobs[0,0,0]) == 1


def test_physical_eight_rank_stream_round_traces_expected_state_abi():
    from tpu_beam_search.beam_runner import make_stream_round_call
    fn = make_stream_round_call(SimpleNamespace(size=8),bins=8,period=1,
        clean_ready_threshold=120,dirty_trigger=1)
    shapes = jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),inputs())
    trace = jax.make_jaxpr(fn,axis_env=[('core',8)])(*shapes)
    assert len(trace.out_avals) == 12


def test_next_round_consumes_published_threshold_and_dedups_old_and_new_records():
    from tpu_beam_search.beam_runner import make_stream_round_call
    args = inputs()
    fn = make_stream_round_call(SimpleNamespace(size=1),bins=8,period=1,
        clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    first,_,_ = fn(*args)
    second,error,jobs = fn(first,*args[1:])
    assert not np.asarray(error).any()
    # Ready reservation leaves the sibling writable. The second batch therefore
    # enters B; cross-sibling duplicates remain until the separate final merge.
    np.testing.assert_array_equal(second.controls[0,:2,0],[2,1])
    np.testing.assert_array_equal(second.a,first.a)
    np.testing.assert_array_equal(np.asarray(second.b)[0,[0,4,6],0],[7,3,1])
    np.testing.assert_array_equal(second.threshold_a[:,0],[1,1])
    np.testing.assert_array_equal(second.epoch[:,0],[0,2,0,0])
    assert int(jobs[0,0,0]) == 1
    third,error,_ = fn(second,*args[1:])
    assert not np.asarray(error).any()
    np.testing.assert_array_equal(third.controls[0,:2,0],[1,1])
    np.testing.assert_array_equal(np.asarray(third.a)[0,[0,4,6],0],[7,3,1])


def test_preexisting_fatal_admits_no_new_s3_records(monkeypatch):
    import tpu_beam_search.beam_runner as runner
    args = inputs()
    state = args[0]._replace(controls=args[0].controls.at[0,7,0].set(1))
    observed = []
    class ReachedCollector(Exception): pass
    def boundary(*call_args):
        observed.append(np.asarray(call_args[5]))
        raise ReachedCollector
    monkeypatch.setattr(runner,'make_stream3_collect_call',lambda *a,**k:boundary)
    fn = runner.make_stream_round_call(SimpleNamespace(size=1),bins=8,period=1,
        clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    with pytest.raises(ReachedCollector):
        fn(state,*args[1:])
    np.testing.assert_array_equal(observed[0],[0])
