from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_beam_s4_service import fixture


def inputs():
    abc,ha,hb,active,epoch = fixture()
    pair = (*abc,ha,hb,active)
    slots = (jnp.zeros((2,128),jnp.uint32),jnp.zeros((2,128),jnp.uint32),
             jnp.zeros((1,128),jnp.uint32))
    beam = jnp.zeros((2,128),jnp.uint32).at[0,0].set(1)
    return (pair,),jnp.zeros_like(epoch),slots,beam,jnp.zeros((1,),jnp.uint32)


def test_service_commits_before_s5_threshold_publication():
    from tpu_beam_search.beam_s4_s5_service import make_s4_s5_service_call
    fn = make_s4_s5_service_call(SimpleNamespace(size=1),bins=8,period=1,
                                clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    pairs,epoch,slots,error,jobs = fn(*inputs())
    assert int(pairs[0][2][0,0]) == 2
    assert int(pairs[0][5][0,0]) == 1
    assert int(slots[2][0,0]) == 1
    np.testing.assert_array_equal(slots[1][:,0],[1,1])
    np.testing.assert_array_equal(epoch[:,0],[0,1,0,0])
    assert not np.asarray(error).any()
    assert int(jobs[0][0,0]) == 1


def test_late_pair_fatal_blocks_all_local_admission_and_threshold_publication():
    from tpu_beam_search.beam_s4_s5_service import make_s4_s5_service_call
    pairs,epoch,slots,beam,force = inputs()
    broken = list(pairs[0])
    broken[2] = broken[2].at[7,0].set(1)
    fn = make_s4_s5_service_call(SimpleNamespace(size=1),bins=8,period=1,
                                clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    result = fn((pairs[0],tuple(broken)),epoch,slots,beam,force)
    assert int(result[3][0,0]) == 1
    for old,new,job in zip((pairs[0],tuple(broken)),result[0],result[4],strict=True):
        for index in (0,1,3,4,5):
            for got,want in zip(jax.tree.leaves(new[index]),jax.tree.leaves(old[index]),strict=True):
                np.testing.assert_array_equal(got,want)
        assert int(new[2][7,0]) == 1
        assert not np.asarray(job).any()
    for got,want in zip(jax.tree.leaves(result[1:3]),jax.tree.leaves((epoch,slots)),strict=True):
        np.testing.assert_array_equal(got,want)


def test_eight_rank_service_traces_expected_abi():
    from tpu_beam_search.beam_s4_s5_service import make_s4_s5_service_call
    fn = make_s4_s5_service_call(SimpleNamespace(size=8),bins=8,period=1,
                                clean_ready_threshold=120,dirty_trigger=1)
    shapes = jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),inputs())
    traced = jax.make_jaxpr(fn,axis_env=[('core',8)])(*shapes)
    assert len(traced.out_avals) == 14


def test_no_local_job_still_runs_forced_s5_epoch():
    from tpu_beam_search.beam_s4_s5_service import make_s4_s5_service_call
    pairs,epoch,slots,beam,force = inputs()
    pair = list(pairs[0])
    pair[2] = jnp.zeros_like(pair[2]).at[0,0].set(2)
    pair[3] = (pair[3][0].at[0,3].set(2),pair[3][1])
    fn = make_s4_s5_service_call(SimpleNamespace(size=1),bins=8,period=1,
                                clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    result = fn((tuple(pair),),epoch,slots,beam,jnp.ones_like(force))
    assert not np.asarray(result[4][0]).any()
    assert int(result[2][2][0,0]) == 1
    np.testing.assert_array_equal(result[2][1][:,0],[3,1])
    np.testing.assert_array_equal(result[1][:,0],[0,1,0,0])
