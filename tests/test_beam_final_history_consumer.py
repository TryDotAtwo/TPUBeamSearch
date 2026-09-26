from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest


def test_history_epochs_preserve_parent64_and_reject_duplicate_batch():
    from tpu_beam_search.beam_final_history_consumer import make_final_history_consumer
    call=make_final_history_consumer(SimpleNamespace(size=1),move_count=24,interpret=True)
    history=jnp.zeros((2,5,128),jnp.uint32)
    seen=jnp.zeros((1,256),jnp.uint32)
    coverage=jnp.zeros((2,128),jnp.uint32)
    common=jnp.zeros((1,128),jnp.uint32)
    status=jnp.zeros((2,128),jnp.uint32).at[0,0].set(1)
    records=np.zeros((5,128),np.uint32)
    records[:,0]=[0xffffffff,17,23,128,1]
    history,seen,coverage,common=call(history,seen,coverage,jnp.asarray(records),
        status,jnp.array([129],jnp.uint32),common)
    expected=np.zeros((2,5,128),np.uint32)
    expected[1,:,0]=[0xffffffff,17,23,128,1]
    np.testing.assert_array_equal(history,expected)
    assert not np.asarray(common).any()
    records[:,0]=[99,0,2,0,1]
    records[:,1]=[88,0,1,128,1]
    history,seen,coverage,common=call(history,seen,coverage,jnp.asarray(records),
        status.at[0,0].set(2),jnp.array([129],jnp.uint32),common)
    np.testing.assert_array_equal(history,expected)
    assert int(common[0,0])==1


@pytest.mark.parametrize('route,valid',[(24,1),(1<<16,1),(0,0),(0,2)])
def test_invalid_history_metadata_cannot_mark_or_store(route,valid):
    from tpu_beam_search.beam_final_history_consumer import make_final_history_consumer
    call=make_final_history_consumer(SimpleNamespace(size=1),move_count=24,interpret=True)
    records=jnp.zeros((5,128),jnp.uint32).at[2,0].set(route).at[4,0].set(valid)
    history,seen,coverage,common=call(jnp.zeros((1,5,128),jnp.uint32),
        jnp.zeros((1,128),jnp.uint32),jnp.zeros((2,128),jnp.uint32),records,
        jnp.zeros((2,128),jnp.uint32).at[0,0].set(1),
        jnp.array([1],jnp.uint32),jnp.zeros((1,128),jnp.uint32))
    assert not np.asarray(history).any() and not np.asarray(seen).any()
    assert int(common[0,0])==1 and int(coverage[1,0])==1


def test_tiled_history_exports_target_order_without_narrowing_parent():
    from tpu_beam_search.beam_final_history_consumer import pallas_history_tiles_to_soa
    history=np.zeros((2,5,128),np.uint32)
    history[0,:,127]=[1,9,23,127,1]
    history[1,:,0]=[0xffffffff,17,7,128,1]
    actual=pallas_history_tiles_to_soa(jnp.asarray(history),interpret=True)
    expected=np.zeros((5,256),np.uint32)
    expected[:,127]=[1,9,23,127,1]
    expected[:,128]=[0xffffffff,17,7,128,1]
    np.testing.assert_array_equal(actual,expected)
