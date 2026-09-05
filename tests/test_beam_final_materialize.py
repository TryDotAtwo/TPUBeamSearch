import numpy as np
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


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
    requests[1,1] = 1 # Out-of-range high parent word: must not read it.
    wire,errors = run(requests)
    np.testing.assert_array_equal(wire,0)
    assert int(errors[0,0]) == 1
    assert int(errors[1,0]) == 1
