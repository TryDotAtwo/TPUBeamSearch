import jax.numpy as jnp
import numpy as np
import pytest


def fixture(selected=0):
    a = np.zeros((8,128),np.uint32)
    b = a.copy()
    for rows in (a,b):
        rows[6] = 0xffffffff
    records = a if selected == 0 else b
    records[0,:3] = [7,7,9]
    records[4,:3] = [8,3,5]
    records[6,:3] = [2,1,2]
    controls = np.zeros((8,128),np.uint32)
    controls[2+selected,0] = 3
    active = np.zeros((1,128),np.uint32)
    epoch = np.zeros((4,128),np.uint32)
    epoch[0,0] = 4
    ha = tuple(jnp.zeros((1,128),jnp.uint32) for _ in range(2))
    hb = tuple(jnp.full((1,128),19,jnp.uint32) for _ in range(2))
    return tuple(map(jnp.asarray,(a,b,controls))),ha,hb,jnp.asarray(active),jnp.asarray(epoch)


@pytest.mark.parametrize('selected',[0,1])
def test_service_claims_dedups_commits_then_counts_completed_job(selected):
    from tpu_beam_search.beam_s4_service import pallas_service_s4_pair
    abc,ha,hb,active,epoch = fixture(selected)
    result = pallas_service_s4_pair(*abc,ha,hb,active,epoch,jnp.array([7],jnp.uint32),
        bins=8,clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    a,b,controls,na,nb,new_active,new_epoch,job = result
    np.testing.assert_array_equal(np.asarray((a,b)[selected])[[0,4,6],:2],
                                  [[7,9],[3,5],[1,2]])
    np.testing.assert_array_equal((a,b)[1-selected],abc[1-selected])
    assert int(controls[selected,0]) == 2
    assert int(controls[2+selected,0]) == 0
    assert not np.asarray(controls[4:6]).any()
    assert int(new_active[0,selected]) == 1
    want_hist = np.zeros((1,128),np.uint32)
    want_hist[0,1:3] = 1
    np.testing.assert_array_equal(nb[selected],want_hist)
    np.testing.assert_array_equal(nb[1-selected],hb[1-selected])
    np.testing.assert_array_equal(na,ha)
    assert int(new_epoch[0,0]) == 5
    np.testing.assert_array_equal(job[:,0],[1,selected])


@pytest.mark.parametrize('fault',['fatal','busy','counter_full'])
def test_service_does_not_publish_work_when_admission_is_blocked(fault):
    from tpu_beam_search.beam_s4_service import pallas_service_s4_pair
    abc,ha,hb,active,epoch = fixture()
    a,b,controls = abc
    if fault == 'fatal': controls = controls.at[7,0].set(1)
    if fault == 'busy': controls = controls.at[5,0].set(1)
    if fault == 'counter_full': epoch = epoch.at[0,0].set(0xffffffff)
    result = pallas_service_s4_pair(a,b,controls,ha,hb,active,epoch,jnp.array([7],jnp.uint32),
        bins=8,clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    want_control = controls.at[7,0].set(1) if fault == 'counter_full' else controls
    expected = (a,b,want_control,ha,hb,active,epoch,jnp.zeros((2,128),jnp.uint32))
    import jax
    for got,want in zip(jax.tree.leaves(result),jax.tree.leaves(expected),strict=True):
        np.testing.assert_array_equal(got,want)


def test_second_service_preserves_prior_histogram_and_flips_active_version():
    from tpu_beam_search.beam_s4_service import pallas_service_s4_pair
    from tpu_beam_search.beam_s5_epoch_state import pallas_s5_local_request
    abc,ha,hb,active,epoch = fixture(1)
    threshold = jnp.array([7],jnp.uint32)
    options = dict(bins=8,clean_ready_threshold=120,dirty_trigger=1,interpret=True)
    first = pallas_service_s4_pair(*abc,ha,hb,active,epoch,threshold,**options)
    second = pallas_service_s4_pair(*first[:7],threshold,force_clean=True,**options)
    assert int(second[5][0,1]) == 0
    assert int(second[6][0,0]) == 6
    np.testing.assert_array_equal(second[3][1],first[4][1])
    np.testing.assert_array_equal(second[4][1],first[4][1])
    request = pallas_s5_local_request(second[6],jnp.zeros((1,),jnp.uint32),
                                    period=6,interpret=True)
    assert int(request[0,0]) == 1


def test_empty_pair_at_counter_limit_does_not_invent_overflow():
    from tpu_beam_search.beam_s4_service import pallas_service_s4_pair
    abc,ha,hb,active,epoch = fixture()
    a,b,_ = abc
    controls = jnp.zeros((8,128),jnp.uint32)
    epoch = epoch.at[0,0].set(0xffffffff)
    result = pallas_service_s4_pair(a,b,controls,ha,hb,active,epoch,jnp.array([7],jnp.uint32),
        bins=8,clean_ready_threshold=120,dirty_trigger=1,force_clean=True,force_dirty=True,
        interpret=True)
    np.testing.assert_array_equal(result[2],controls)
    np.testing.assert_array_equal(result[6],epoch)
    assert not np.asarray(result[7]).any()
