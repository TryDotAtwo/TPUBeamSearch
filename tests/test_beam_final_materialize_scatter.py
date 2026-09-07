"""Local integration oracle; does not claim physical TPU acceptance."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu


@pytest.mark.parametrize('count',[0,1,127,128,129])
def test_materialize_then_scatter_preserves_unselected_frontier(count):
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    parents = (np.arange(7*128).reshape(7,128)%256).astype(np.uint8)
    generators = np.arange(127,-1,-1,dtype=np.int32)[None,:]
    requests = np.zeros((4,256),np.uint32)
    requests[0,:count] = np.arange(count)%7
    targets = (np.arange(count)*3)%256  # Unique, including non-eight-aligned IDs.
    requests[2,:count] = targets
    frontier = np.full((256,128),173,np.uint8)
    expected = frontier.copy()
    for i,target in enumerate(targets):
        expected[target,:120] = parents[i%7,generators[0,:120]]
        expected[target,120:] = 0
    mode = pltpu.InterpretParams(detect_races=True)
    wire,validation = pallas_materialize_final(jnp.asarray(parents),jnp.asarray(generators),
        jnp.asarray(requests),jnp.array([count],jnp.uint32),jnp.array([256],jnp.uint32),
        state_len=120,interpret=mode)
    actual,scatter_errors = pallas_scatter_final_responses(jnp.asarray(frontier),wire,
        jnp.array([count],jnp.uint32),state_len=120,interpret=mode)
    assert int(validation[0,0]) == int(scatter_errors[0,0]) == 0
    np.testing.assert_array_equal(actual,expected)
