import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('count', [129, 0xffffffff])
def test_overflow_rejects_entire_batch(count):
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    parents = jnp.ones((1,128), jnp.uint8)
    generators = jnp.tile(jnp.arange(128,dtype=jnp.int32),(24,1))
    requests = jnp.zeros((4,128),jnp.uint32)
    n = jnp.array([count],jnp.uint32)
    wire, errors = pallas_materialize_final(parents,generators,requests,n,
        jnp.array([1],jnp.uint32),state_len=120,interpret=True)
    assert int(errors[0,0]) > 0
    assert not np.asarray(wire).any()
    frontier = jnp.full((1,128),99,jnp.uint8)
    result, errors = pallas_scatter_final_responses(frontier,jnp.zeros((128,128),jnp.uint8),
        n,state_len=120,interpret=True)
    assert int(errors[0,0]) > 0
    np.testing.assert_array_equal(result,frontier)


@pytest.mark.parametrize('count', [0,128])
def test_boundary_counts_remain_valid(count):
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    parents = jnp.ones((1,128),jnp.uint8)
    generators = jnp.tile(jnp.arange(128,dtype=jnp.int32),(24,1))
    requests = jnp.zeros((4,128),jnp.uint32).at[2].set(jnp.arange(128,dtype=jnp.uint32))
    n = jnp.array([count],jnp.uint32)
    wire, errors = pallas_materialize_final(parents,generators,requests,n,
        jnp.array([128],jnp.uint32),state_len=120,interpret=True)
    assert int(errors[0,0]) == 0
    frontier = jnp.full((128,128),99,jnp.uint8)
    result, errors = pallas_scatter_final_responses(frontier,wire,n,state_len=120,interpret=True)
    assert int(errors[0,0]) == 0
    expected = np.full((128,128),99,np.uint8)
    expected[:count,:120] = 1
    expected[:count,120:] = 0
    np.testing.assert_array_equal(result,expected)
