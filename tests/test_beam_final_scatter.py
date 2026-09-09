import numpy as np
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu


@pytest.mark.parametrize('count',[0,1])
@pytest.mark.parametrize('flag',[0,1,0x80000000])
def test_scatter_prior_error_preserves_all_frontier_bytes(count,flag):
    from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses
    frontier = np.arange(4*128,dtype=np.uint8).reshape(4,128)
    wire = np.zeros((128,128),np.uint8)
    wire[0,:120] = 7
    wire[0,120] = 2
    prior = jnp.zeros((1,128),jnp.uint32).at[0,0].set(flag)
    actual,errors = pallas_scatter_final_responses(jnp.asarray(frontier.copy()),
        jnp.asarray(wire),jnp.array([count],jnp.uint32),state_len=120,
        prior_error=prior,interpret=pltpu.InterpretParams(detect_races=True))
    want = frontier.copy()
    if not flag and count:
        want[2,:120] = 7
        want[2,120:] = 0
    np.testing.assert_array_equal(actual,want)
    assert int(errors[0,0]) == int(flag != 0)
    assert int(errors[1,0]) == (0 if flag else 0xffffffff)


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
