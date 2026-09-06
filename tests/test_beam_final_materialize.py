import numpy as np
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def test_materialize_output_avoids_illegal_single_row_pipeline_block():
    """Structural regression for the V1 TPU rejection, not TPU acceptance."""
    import jax
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    args = (jnp.zeros((2,128),jnp.uint8),
            jnp.tile(jnp.arange(128,dtype=jnp.int32),(2,1)),
            jnp.zeros((4,128),jnp.uint32),jnp.zeros((1,),jnp.uint32),
            jnp.ones((1,),jnp.uint32))
    traced = jax.make_jaxpr(lambda *xs: pallas_materialize_final(
        *xs,state_len=120,interpret=True))(*args)
    calls = [eq for eq in traced.jaxpr.eqns if eq.primitive.name == 'pallas_call']
    mapping = calls[-1].params['grid_mapping'].block_mappings[-1]
    sizes = tuple(getattr(dim,'block_size',dim) for dim in mapping.block_shape)
    whole = calls[-1].outvars[0].aval.shape
    assert sizes[-2] == whole[-2] or sizes[-2] % 8 == 0, sizes


def test_materialize_dma_record_axis_is_outside_minor_tiled_dimensions():
    """V2 regression: arbitrary parent IDs must not slice a tiled row axis."""
    import jax
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    args = (jnp.zeros((7,128),jnp.uint8),
            jnp.tile(jnp.arange(128,dtype=jnp.int32),(24,1)),
            jnp.zeros((4,128),jnp.uint32),jnp.zeros((1,),jnp.uint32),
            jnp.ones((1,),jnp.uint32))
    traced = jax.make_jaxpr(lambda *xs: pallas_materialize_final(
        *xs,state_len=120,interpret=True))(*args)
    call = [eq for eq in traced.jaxpr.eqns if eq.primitive.name == 'pallas_call'][-1]
    assert call.invars[0].aval.shape == (7,1,128)
    assert call.outvars[0].aval.shape == (128,1,128)


def test_materialize_valid_and_reject_entire_invalid_batch():
    from tpu_beam_search.beam_final_materialize import pallas_materialize_final
    parents = np.zeros((2,128),np.uint8)
    parents[0,:3],parents[1,:3] = [0,1,2],[2,0,1]
    g = np.tile(np.arange(128,dtype=np.int32),(2,1))
    g[1,:3] = [1,2,0]
    requests = np.zeros((4,128),np.uint32)
    requests[0,:2] = [1,0]
    requests[2,:2] = [7,5]
    requests[3,0] = 1<<16
    def run(r):
        return pallas_materialize_final(*map(jnp.asarray,(parents,g,r)),
            jnp.array([2],jnp.uint32),jnp.array([8],jnp.uint32),state_len=120,
            interpret=pltpu.InterpretParams(detect_races=True))
    wire,errors = run(requests)
    assert int(errors[0,0]) == 0
    want = np.zeros((128,128),np.uint8)
    want[0,:120],want[1,:120] = parents[1,g[1]][:120],parents[0,:120]
    want[0,120],want[1,120] = 7,5
    np.testing.assert_array_equal(wire,want)
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    frontier,scatter_errors = pallas_scatter_final_responses(jnp.zeros((8,128),jnp.uint8),
        wire,jnp.array([2],jnp.uint32),state_len=120,
        interpret=pltpu.InterpretParams(detect_races=True))
    expected_frontier = np.zeros((8,128),np.uint8)
    expected_frontier[7,:120],expected_frontier[5,:120] = want[0,:120],want[1,:120]
    np.testing.assert_array_equal(frontier,expected_frontier)
    assert int(scatter_errors[0,0]) == 0
    requests[1,1] = 1 # Out-of-range high parent word: must not read it.
    wire,errors = run(requests)
    np.testing.assert_array_equal(wire,0)
    assert int(errors[0,0]) == 1
    assert int(errors[1,0]) == 1
