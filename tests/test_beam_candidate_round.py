from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from test_beam_stream_round import inputs
from test_beam_k1_neighborhood import inputs as puzzle_inputs


def fixture():
    from tpu_beam_search.beam_candidate_round import CandidateRoundState
    stream,*_,beam,force,neutral = inputs()
    state = CandidateRoundState(stream,jnp.zeros((10,128),jnp.uint32),
        jnp.zeros((4,128),jnp.uint32),jnp.zeros((1,128),jnp.uint32),
        jnp.zeros((1,128),jnp.uint32))
    central,g,z = puzzle_inputs()
    parents = np.tile(central,(2,1))
    parents[0,:3] = [1,0,2]
    return state,tuple(map(jnp.asarray,(parents,g,central,z)))+(
        jnp.array([2],jnp.uint32),jnp.ones((1,128),jnp.uint32),
        jnp.array([0xffffffff,2],jnp.uint32),jnp.array([0],jnp.uint32),
        jnp.array([8],jnp.uint32),beam,force,neutral)


def factory(mesh=1,interpret=True):
    from tpu_beam_search.beam_candidate_round import make_candidate_round_call
    return make_candidate_round_call(SimpleNamespace(size=mesh),bins=8,period=1,
        clean_ready_threshold=120,dirty_trigger=1,stop_on_found=True,interpret=interpret)


def test_s2_solved_and_streams_round_then_stop_blocks_new_parents():
    state,args = fixture()
    call = factory()
    first,jobs = call(state,*args)
    assert int(first.error[0,0]) == 0 and int(first.stop[0,0]) == 1
    assert int(first.solved_control[0,0]) == 1
    assert int(first.solved_control[2,0]) == 1
    assert int(jobs[0,0,0]) == 1
    assert int(first.streams.controls[0,0,0]) > 0
    second,_ = call(first,*args)
    np.testing.assert_array_equal(second.solved,first.solved)
    np.testing.assert_array_equal(second.solved_control,first.solved_control)
    np.testing.assert_array_equal(second.streams.a,first.streams.a)
    np.testing.assert_array_equal(second.streams.b,first.streams.b)
    assert int(second.stop[0,0]) == 1 and int(second.error[0,0]) == 0


def test_identity_failure_propagates_fatal_without_candidate_or_solution_write():
    state,args = fixture()
    args = list(args)
    args[6] = jnp.array([0xffffffff,0xffffffff],jnp.uint32)
    out,jobs = factory()(state,*args)
    assert int(out.error[0,0]) == 1 and int(out.stop[0,0]) == 1
    np.testing.assert_array_equal(out.solved,state.solved)
    np.testing.assert_array_equal(out.streams.a,state.streams.a)
    np.testing.assert_array_equal(out.streams.b,state.streams.b)
    assert not np.asarray(jobs).any()


def test_eight_rank_candidate_round_traces_without_host_counts():
    state,args = fixture()
    trace = jax.make_jaxpr(factory(8,False),axis_env=[('core',8)])(state,*args)
    assert len(trace.out_avals) == 15
