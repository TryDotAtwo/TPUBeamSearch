from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tpu_beam_search.beam_stream2_batch import Stream2Batch


def fixture():
    words = np.zeros((8,128),np.uint32)
    words[4,:2], words[5,:2] = [0xffffffff,0], [2,3]
    words[7,:2] = [1,0]
    solution = np.zeros((4,128),np.uint32)
    solution[:,:2] = np.array([[10,20],[11,21],[12,22],[13,23]],np.uint32)
    flags = np.zeros((1,128),np.uint32)
    flags[0,:2] = 1
    ids = np.zeros_like(flags)
    ids[0,:2] = [0,7]
    batch = Stream2Batch(*map(jnp.asarray,(words,np.zeros((1,128),np.uint32),
        np.array([2],np.uint32),np.zeros((2,128),np.uint32),flags,solution,ids)))
    return batch, jnp.full((10,128),99,jnp.uint32), jnp.zeros((4,128),jnp.uint32)


def test_solved_service_stores_projected_hash_parent_suffix_and_common_stop():
    from tpu_beam_search.beam_solved_service import make_solved_batch_service
    batch,arena,control = fixture()
    call = make_solved_batch_service(SimpleNamespace(size=1),local_rank=3,
                                    stop_on_found=True,interpret=True)
    out,ctl,error,stop = call(arena,control,batch,jnp.array([8],jnp.uint32),
                            jnp.zeros((1,128),jnp.uint32))
    expected = np.full((10,128),99,np.uint32)
    expected[:,:2] = np.array([[10,20],[11,21],[12,22],[13,23],
        [0xffffffff,0],[2,3],[0,0],[0x30301,0x30300],[8,8],[0,7]],np.uint32)
    np.testing.assert_array_equal(out,expected)
    np.testing.assert_array_equal(np.asarray(ctl)[:,0],[2,0,1,1])
    assert int(error[0,0]) == 0 and int(stop[0,0]) == 1


@pytest.mark.parametrize('fault',['counter','batch','prior'])
def test_solved_service_rejects_failed_batch_without_losing_old_records(fault):
    from tpu_beam_search.beam_solved_service import make_solved_batch_service
    batch,arena,control = fixture()
    prior = jnp.zeros((1,128),jnp.uint32)
    if fault == 'counter': control = control.at[0,0].set(0xffffffff)
    if fault == 'batch': batch = batch._replace(error=batch.error.at[0,0].set(1))
    if fault == 'prior': prior = prior.at[0,0].set(1)
    call = make_solved_batch_service(SimpleNamespace(size=1),local_rank=0,
                                    stop_on_found=False,interpret=True)
    out,ctl,error,stop = call(arena,control,batch,jnp.array([8],jnp.uint32),prior)
    np.testing.assert_array_equal(out,arena)
    np.testing.assert_array_equal(ctl,control)
    assert int(error[0,0]) == 1 and int(stop[0,0]) == 1


def test_eight_rank_solved_service_traces_control_abi():
    from tpu_beam_search.beam_solved_service import make_solved_batch_service
    batch,arena,control = fixture()
    call = make_solved_batch_service(SimpleNamespace(size=8),local_rank=0,
                                    stop_on_found=True)
    trace = jax.make_jaxpr(call,axis_env=[('core',8)])(
        arena,control,batch,jnp.array([8],jnp.uint32),jnp.zeros((1,128),jnp.uint32))
    assert tuple(a.shape for a in trace.out_avals) == ((10,128),(4,128),(1,128),(1,128))


def test_empty_saturated_counter_is_not_wrap_and_inflight_stop_is_preserved():
    from tpu_beam_search.beam_solved_service import make_solved_batch_service
    batch,arena,control = fixture()
    batch = batch._replace(found=jnp.zeros_like(batch.found))
    control = control.at[0,0].set(0xffffffff).at[3,0].set(1)
    call = make_solved_batch_service(SimpleNamespace(size=1),local_rank=0,
                                    stop_on_found=False,interpret=True)
    out,ctl,error,stop = call(arena,control,batch,jnp.array([8],jnp.uint32),
                            jnp.zeros((1,128),jnp.uint32))
    np.testing.assert_array_equal(out,arena)
    np.testing.assert_array_equal(ctl,control)
    assert int(error[0,0]) == 0 and int(stop[0,0]) == 1


def test_storage_overflow_counts_inflight_hits_without_overwriting_old_results():
    from tpu_beam_search.beam_solved_service import make_solved_batch_service
    batch,arena,control = fixture()
    control = control.at[0,0].set(127).at[2,0].set(1).at[3,0].set(1)
    call = make_solved_batch_service(SimpleNamespace(size=1),local_rank=0,
                                    stop_on_found=False,interpret=True)
    out,ctl,error,stop = call(arena,control,batch,jnp.array([8],jnp.uint32),
                            jnp.zeros((1,128),jnp.uint32))
    np.testing.assert_array_equal(np.asarray(out)[:,:127],np.asarray(arena)[:,:127])
    np.testing.assert_array_equal(np.asarray(out)[:,127],
        [10,11,12,13,0xffffffff,2,0,1,8,0])
    np.testing.assert_array_equal(np.asarray(ctl)[:,0],[129,1,1,1])
    assert int(error[0,0]) == 0 and int(stop[0,0]) == 1
