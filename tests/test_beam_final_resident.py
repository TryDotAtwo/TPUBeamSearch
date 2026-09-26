import jax.numpy as jnp
import numpy as np
import pytest


def resident():
    a = np.zeros((2,8,128),np.uint32)
    b = np.zeros_like(a)
    a[:,6],b[:,6] = 0xffffffff,0xffffffff
    # Same hash in A0 and B0; distinct parent history must survive final view.
    a[:,0,0], b[:,0,0] = [7,9], [7,11]
    a[:,4,0], b[:,4,0] = [10,30], [20,40]
    a[:,6,0], b[:,6,0] = 5,5
    ctl = np.zeros((2,8,128),np.uint32)
    ctl[:,:2,0] = 1
    return tuple(map(jnp.asarray,(a,b,ctl,np.zeros((1,128),np.uint32))))


@pytest.mark.parametrize('cap', [1,4])
def test_physical_sibling_traversal_preserves_duplicates_and_cuda_tie_order(cap):
    from tpu_beam_search.beam_final_resident import pallas_final_resident_view
    from tpu_beam_search.beam_final_phase import pallas_final_phase_masks
    from tpu_beam_search.beam_final_scan import pallas_final_phase_scan
    from tpu_beam_search.beam_final_prefix import pallas_final_prefixes
    from tpu_beam_search.beam_final_cap import pallas_final_cap
    from tpu_beam_search.beam_final_indices import pallas_final_indices
    meta,clean,error = pallas_final_resident_view(*resident(),interpret=True)
    np.testing.assert_array_equal(np.asarray(meta)[:,4,0],[10,20,30,40])
    np.testing.assert_array_equal(np.asarray(meta)[:,0,0],[7,7,9,11])
    assert int(error[0,0]) == 0
    masks = pallas_final_phase_masks(meta[:,6,:],clean,jnp.array([5],jnp.uint32),interpret=True)
    ordinal,counts = pallas_final_phase_scan(masks,interpret=True)
    np.testing.assert_array_equal(np.asarray(counts)[:,0],[0,4])
    bases,totals = pallas_final_prefixes(counts,world_size=1,interpret=True)
    beam = jnp.zeros((2,128),jnp.uint32).at[0,0].set(cap)
    keep,cap_error = pallas_final_cap(totals,beam,interpret=True)
    indices,valid = pallas_final_indices(ordinal,bases,keep,cap_error,rank=0,interpret=True)
    selected = np.flatnonzero(np.asarray(valid)[1,:,0])
    np.testing.assert_array_equal(np.asarray(meta)[selected,4,0],[10,20,30,40][:cap])
    np.testing.assert_array_equal(np.asarray(indices)[1,0,selected,0],np.arange(cap))


@pytest.mark.parametrize('fault',['dirty','busy','fatal','count','prior'])
def test_unfrozen_or_failed_resident_state_cannot_enter_final_selection(fault):
    from tpu_beam_search.beam_final_resident import pallas_final_resident_view
    a,b,c,error = resident()
    if fault == 'dirty': c = c.at[1,3,0].set(1)
    if fault == 'busy': c = c.at[1,5,0].set(1)
    if fault == 'fatal': c = c.at[1,7,0].set(1)
    if fault == 'count': c = c.at[1,1,0].set(129)
    if fault == 'prior': error = error.at[0,0].set(1)
    _,clean,error = pallas_final_resident_view(a,b,c,error,interpret=True)
    assert not np.asarray(clean).any() and int(error[0,0]) == 1
