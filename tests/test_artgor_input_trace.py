import jax.numpy as jnp
import numpy as np


def test_trace_preserves_raw_mean_but_centers_rounded_values():
    import benchmarks.artgor_input_trace as trace
    raw = jnp.tile(jnp.array([[-1., 1. + 1/256]], jnp.float32), (8, 64))
    scale = jnp.full((128,), 256., jnp.bfloat16)
    bias = jnp.full((128,), -255., jnp.bfloat16)
    for pallas in (False, True):
        result = trace.input_trace(raw, scale, bias, epsilon=1e-5,
                                   pallas=pallas, interpret=True, bm=8)
        np.testing.assert_array_equal(result[0], np.full((8,128), .25))
        np.testing.assert_array_equal(result[1], np.full((8,128), 1/512))
        np.testing.assert_array_equal(result[2][0,:2], [-1-1/512, 1-1/512])
        np.testing.assert_array_equal(result[-1][0,:2], [0., .5])


def test_mismatch_capture_keeps_all_coordinates_and_only_affected_rows(tmp_path):
    import benchmarks.artgor_input_trace as trace
    raw = np.arange(12, dtype=np.float32).reshape(3,4)
    ref = np.zeros((3,4), np.float32)
    candidate = ref.copy()
    candidate[1,0] = 1
    candidate[1,3] = 2
    path = tmp_path / 'witness.npz'
    trace.save_mismatch_rows(path, raw, ref, candidate)
    with np.load(path) as data:
        np.testing.assert_array_equal(data['coordinates'], [[1,0],[1,3]])
        np.testing.assert_array_equal(data['row_ids'], [1])
        np.testing.assert_array_equal(data['raw'], raw[1:2])
        np.testing.assert_array_equal(data['candidate'], candidate[1:2])
