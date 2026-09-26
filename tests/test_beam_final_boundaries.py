import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('world,keep',[(1,0),(1,17),(2,1),(3,5),(8,19),
    (8,0xffffffff),(8,0x100000005),(127,127*0xffffffff),
    (3,3*0xffffffff),(1,0x100000000),(8,8*0xffffffff+1),(127,0xffffffffffffffff)])
def test_exact_boundaries_and_local_capacity_gate(world,keep):
    from tpu_beam_search.beam_final_boundaries import pallas_final_boundaries
    pair = np.zeros((2,128),np.uint32)
    pair[:,0] = keep&0xffffffff,keep>>32
    bounds,counts,error = pallas_final_boundaries(jnp.asarray(pair),
        jnp.zeros((1,128),jnp.uint32),world_size=world,interpret=True)
    rejected = (keep+world-1)//world > 0xffffffff
    assert int(error[0,0]) == int(rejected)
    want_bounds = np.zeros((2,128),np.uint32)
    want_counts = np.zeros((1,128),np.uint32)
    if not rejected:
        exact = [(rank*keep+world-1)//world for rank in range(world+1)]
        for rank,value in enumerate(exact):
            want_bounds[:,rank] = value&0xffffffff,value>>32
        want_counts[0,:world] = np.diff(exact)
    np.testing.assert_array_equal(bounds,want_bounds)
    np.testing.assert_array_equal(counts,want_counts)


def test_prior_selection_error_cannot_publish_balance_intervals():
    from tpu_beam_search.beam_final_boundaries import pallas_final_boundaries
    pair = jnp.zeros((2,128),jnp.uint32).at[0,0].set(19)
    bounds,counts,error = pallas_final_boundaries(pair,
        jnp.zeros((1,128),jnp.uint32).at[0,0].set(1),world_size=8,interpret=True)
    assert not np.asarray(bounds).any() and not np.asarray(counts).any()
    assert int(error[0,0]) == 1
