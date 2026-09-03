import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('order,expected', [
    ('lanes_serial', 1/256), ('lanes_tree', 1/256),
    ('tiles_serial', 0.), ('tiles_tree', 0.),
])
def test_reduction_order_preserves_specified_cancellation(order, expected):
    import benchmarks.artgor_input_trace as trace
    raw = np.zeros((8,256), np.float32)
    raw[:,0], raw[:,1], raw[:,128] = 2**24, 1, -(2**24)
    mean = trace.mean_buffer(jnp.asarray(raw), pallas=True, interpret=True, bm=8, order=order)
    np.testing.assert_array_equal(np.asarray(mean, np.float32), np.full((8,256), expected))


def test_unknown_reduction_order_is_rejected():
    import benchmarks.artgor_input_trace as trace
    with pytest.raises(ValueError, match='order'):
        trace.mean_buffer(jnp.zeros((8,128), jnp.float32), pallas=True, order='unknown')


@pytest.mark.parametrize('family', ['lanes', 'tiles'])
def test_serial_and_tree_have_distinct_four_part_rounding(family):
    import benchmarks.artgor_input_trace as trace
    raw = np.zeros((8,512), np.float32)
    raw[:,[0,128,256,384]] = [2**25, 1, -(2**25), 1]
    serial = trace.mean_buffer(jnp.asarray(raw), pallas=True, interpret=True, bm=8, order=family+'_serial')
    tree = trace.mean_buffer(jnp.asarray(raw), pallas=True, interpret=True, bm=8, order=family+'_tree')
    np.testing.assert_array_equal(np.asarray(serial, np.float32), np.full((8,512), 1/512))
    np.testing.assert_array_equal(np.asarray(tree, np.float32), np.zeros((8,512)))


def test_materialized_mean_and_remainder_honor_external_bf16_mean():
    import benchmarks.artgor_input_trace as trace
    raw = jnp.tile(jnp.array([[-1., 1. + 1/256]], jnp.float32), (8, 64))
    scale = jnp.full((128,), 256., jnp.bfloat16)
    bias = jnp.full((128,), -255., jnp.bfloat16)
    for pallas in (False, True):
        mean = trace.mean_buffer(raw, pallas=pallas, interpret=True, bm=8)
        assert mean.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(mean, np.float32), np.full((8,128), 1/512))
        out = trace.external_mean_ln(raw, mean, scale, bias, interpret=True, bm=8)
        np.testing.assert_array_equal(out[0,:2], [0., .5])
        changed = trace.external_mean_ln(raw, jnp.zeros_like(mean), scale, bias, interpret=True, bm=8)
        np.testing.assert_array_equal(changed[0,:2], [0., 1.])


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
