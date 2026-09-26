from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_beam_final_resident import resident


@pytest.mark.parametrize('fault',[None,'dirty','too_many_less'])
def test_final_selection_runs_real_phases_cap_and_compaction(fault):
    from tpu_beam_search.beam_final_selection import make_final_selection_call
    a,b,controls,prior = resident()
    b = b.at[0,6,0].set(2)
    if fault == 'dirty': controls = controls.at[1,2,0].set(1)
    beam = jnp.zeros((2,128),jnp.uint32).at[0,0].set(0 if fault == 'too_many_less' else 3)
    packed,keep,error,counts = make_final_selection_call(
        SimpleNamespace(size=1),interpret=True)(
            a,b,controls,jnp.array([5],jnp.uint32),beam,prior)
    assert int(error[0,0]) == int(fault is not None)
    if fault:
        assert not np.asarray(packed).any()
    else:
        np.testing.assert_array_equal(np.asarray(packed)[4,:3],[20,10,30])
        np.testing.assert_array_equal(np.asarray(packed)[8,:3],[0,1,2])
        np.testing.assert_array_equal(np.asarray(packed)[10,:3],[1,1,1])
        assert not np.asarray(packed)[:,3:].any()
        np.testing.assert_array_equal(np.asarray(counts)[:,0],[1,3])
        np.testing.assert_array_equal(np.asarray(keep)[:,0],[3,0])


def test_eight_rank_final_selection_traces_dynamic_rank_prefix_consumption():
    from tpu_beam_search.beam_final_selection import make_final_selection_call
    a,b,controls,prior = resident()
    fn = make_final_selection_call(SimpleNamespace(size=8))
    trace = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        a,b,controls,jnp.array([5],jnp.uint32),jnp.zeros((2,128),jnp.uint32),prior)
    assert tuple(x.shape for x in trace.out_avals) == ((11,1024),(2,128),(1,128),(2,128))
