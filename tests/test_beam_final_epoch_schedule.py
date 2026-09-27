from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('request_count,history_count,want',[(0,0,0),(128,1,1),(129,128,2),(1,257,3)])
def test_schedule_covers_both_channels_including_partial_tail(request_count,history_count,want):
    from tpu_beam_search.beam_final_epoch_schedule import make_final_epoch_schedule
    ri=jnp.zeros((3,128),jnp.uint32).at[1,0].set(request_count).at[1,7].set(0xffffffff)
    hi=jnp.zeros((3,128),jnp.uint32).at[1,0].set(history_count)
    rounds,error=make_final_epoch_schedule(SimpleNamespace(size=1),request_capacity=256,
        history_capacity=512,interpret=True)(ri,hi,jnp.zeros((1,128),jnp.uint32))
    assert int(rounds[0])==want and not np.asarray(error).any()


@pytest.mark.parametrize('fault',['offset','count','route','prior'])
def test_bad_interval_rejects_schedule_instead_of_running_unbounded_epochs(fault):
    from tpu_beam_search.beam_final_epoch_schedule import make_final_epoch_schedule
    ri=jnp.zeros((3,128),jnp.uint32).at[1,0].set(1)
    hi=jnp.zeros((3,128),jnp.uint32)
    prior=jnp.zeros((1,128),jnp.uint32)
    if fault=='offset': ri=ri.at[0,0].set(256)
    if fault=='count': ri=ri.at[1,0].set(0xffffffff)
    if fault=='route': hi=hi.at[2,0].set(1)
    if fault=='prior': prior=prior.at[0,0].set(7)
    rounds,error=make_final_epoch_schedule(SimpleNamespace(size=1),request_capacity=256,
        history_capacity=512,interpret=True)(ri,hi,prior)
    assert int(rounds[0])==0 and int(error[0,0])==1
