import numpy as np
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def test_scatter_dma_record_axis_avoids_v8_tiled_row_slice():
    """Structural V8 regression; physical compilation remains a separate gate."""
    import jax
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    traced=jax.make_jaxpr(lambda f,w,c: pallas_scatter_final_responses(
        f,w,c,state_len=120,interpret=True))(
            jnp.zeros((256,128),jnp.uint8),jnp.zeros((128,128),jnp.uint8),
            jnp.ones((1,),jnp.uint32))
    call=[eq for eq in traced.jaxpr.eqns if eq.primitive.name=='pallas_call'][-1]
    assert call.invars[0].aval.shape==(256,1,128)
    assert call.invars[1].aval.shape==(128,1,128)
    assert call.outvars[0].aval.shape==(256,1,128)


def test_scatter_response_reorders_and_rejects_out_of_capacity_batch():
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    wire = np.zeros((128,128),np.uint8)
    wire[0,:120],wire[1,:120],wire[2,:120] = 1,2,3
    wire[:3,120] = [7,1,5]
    frontier = np.full((8,128),99,np.uint8)
    def run(w):
        return pallas_scatter_final_responses(jnp.asarray(frontier.copy()),jnp.asarray(w),
            jnp.array([3],jnp.uint32),state_len=120,
            interpret=pltpu.InterpretParams(detect_races=True))
    actual,errors = run(wire)
    want = frontier.copy()
    for row,target in enumerate((7,1,5)):
        want[target] = wire[row]
        want[target,120:] = 0
    np.testing.assert_array_equal(actual,want)
    assert int(errors[0,0]) == 0
    wire[1,120] = 8
    actual,errors = run(wire)
    np.testing.assert_array_equal(actual,frontier)
    assert int(errors[0,0]) == 1
