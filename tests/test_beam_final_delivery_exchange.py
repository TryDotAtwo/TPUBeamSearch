from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('bad_source',[False,True])
def test_paired_delivery_epochs_route_both_channels_and_share_errors(bad_source):
    from tpu_beam_search.beam_final_delivery import pallas_final_delivery_plan
    from tpu_beam_search.beam_final_delivery_exchange import (
        pallas_prepare_delivery_exchange,make_delivery_chunk_call)
    packed = np.zeros((11,256),np.uint32)
    packed[4,:129] = np.arange(129,dtype=np.uint32)
    packed[7,:129] = (2<<16) if bad_source else 0
    packed[8,:129] = np.arange(129,dtype=np.uint32)
    packed[10,:129] = 1
    plan = pallas_final_delivery_plan(jnp.asarray(packed),
        jnp.zeros((2,128),jnp.uint32).at[0,0].set(129),
        jnp.zeros((1,128),jnp.uint32),world_size=1,interpret=True)
    prepared = pallas_prepare_delivery_exchange(plan,world_size=1,interpret=True)
    call = make_delivery_chunk_call(SimpleNamespace(size=1),interpret=True)
    for epoch,length in ((0,128),(1,1),(2,0)):
        requests,counts,history,status,error = call(*prepared,jnp.array([epoch],jnp.uint32))
        assert int(error[0,0]) == int(bad_source)
        amount = 0 if bad_source else length
        assert int(counts[0,0,0]) == amount
        np.testing.assert_array_equal(np.asarray(status)[:,0],[amount,int(bad_source)])
        if amount:
            ids = np.arange(epoch*128,epoch*128+amount,dtype=np.uint32)
            np.testing.assert_array_equal(np.asarray(requests)[0,0,:amount],ids)
            np.testing.assert_array_equal(np.asarray(history)[0,:amount],ids)
            np.testing.assert_array_equal(np.asarray(history)[3,:amount],ids)
        assert not np.asarray(history)[:,amount:].any()


def test_eight_rank_delivery_chunk_traces_both_routes():
    from tpu_beam_search.beam_final_delivery_exchange import make_delivery_chunk_call
    fn = make_delivery_chunk_call(SimpleNamespace(size=8))
    shape = lambda s:jax.ShapeDtypeStruct(s,jnp.uint32)
    trace = jax.make_jaxpr(fn,axis_env=[('core',8)])(
        shape((7,1024)),shape((3,128)),shape((8,1024)),shape((3,128)),
        shape((1,128)),shape((1,)))
    assert tuple(x.shape for x in trace.out_avals) == (
        (8,4,128),(8,1,128),(5,1024),(2,128),(1,128))


def test_late_history_error_invalidates_already_received_requests():
    from tpu_beam_search.beam_final_delivery_exchange import make_delivery_chunk_call
    requests = jnp.zeros((7,128),jnp.uint32).at[0,0].set(41)
    intervals = jnp.zeros((3,128),jnp.uint32).at[1,0].set(1)
    history = jnp.zeros((8,128),jnp.uint32)
    bad_history = intervals.at[2,0].set(1)
    received,counts,hist,status,error = make_delivery_chunk_call(
        SimpleNamespace(size=1),interpret=True)(requests,intervals,history,
            bad_history,jnp.zeros((1,128),jnp.uint32),jnp.array([0],jnp.uint32))
    # Payload reception alone is not acceptance: the common final error gates it.
    assert int(received[0,0,0]) == 41 and int(counts[0,0,0]) == 1
    assert int(error[0,0]) == 1 and int(status[1,0]) == 1
    assert not np.asarray(hist).any()
