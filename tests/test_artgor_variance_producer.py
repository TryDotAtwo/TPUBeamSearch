import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('arithmetic', ['fp32', 'original'])
def test_separate_mean_vector_keeps_per_row_statistics(transposed, arithmetic):
    from benchmarks.artgor_variance_producer import separate_invstd
    dense = jnp.asarray([[1, 3], [6, 6], [3, 7]], jnp.bfloat16)
    mean = jnp.asarray([0, 4, 5], jnp.bfloat16)
    result = separate_invstd(dense.T if transposed else dense, mean,
        transposed=transposed, arithmetic=arithmetic, epsilon=4)
    np.testing.assert_array_equal(result, jnp.asarray([1/3, 1/np.sqrt(8), 1/np.sqrt(8)], jnp.bfloat16))


def test_two_argument_collector_preserves_mean_pairing_and_device_order():
    from benchmarks.artgor_variance_producer import collect_separate
    dense = np.arange(32).reshape(16, 2)
    mean = np.arange(16)*100
    seen = []
    def call(d,m):
        seen.append((d.copy(),m.copy()))
        return d[:,0] + m
    result = collect_separate(dense,mean,call,devices=2,chunk_rows=2)
    np.testing.assert_array_equal(seen[0][0][:,0],[0,2,16,18])
    np.testing.assert_array_equal(seen[0][1],[0,100,800,900])
    np.testing.assert_array_equal(result,dense[:,0]+mean)


@pytest.mark.parametrize('arithmetic', ['fp32', 'original'])
def test_fixed_mean_producer_does_not_recompute_mean(arithmetic):
    from benchmarks.artgor_variance_producer import centered_squares, reduce_invstd, fused_invstd
    dense = jnp.asarray([[1, 3]], jnp.bfloat16)
    mean = jnp.zeros_like(dense)
    squares = centered_squares(dense, mean, arithmetic=arithmetic)
    np.testing.assert_array_equal(squares, [[1, 9]])
    # Fixed mean0 gives variance5, epsilon4 gives invstd1/3.
    expected = np.asarray(jnp.asarray([[1/3]], jnp.bfloat16))
    np.testing.assert_array_equal(reduce_invstd(squares, arithmetic=arithmetic, epsilon=4), expected)
    np.testing.assert_array_equal(fused_invstd(dense, mean, arithmetic=arithmetic, epsilon=4), expected)


def test_centered_square_precision_modes_are_distinct():
    from benchmarks.artgor_variance_producer import centered_squares
    dense = jnp.asarray([[1]], jnp.bfloat16)
    mean = jnp.asarray([[0.00390625]], jnp.bfloat16)
    fp = centered_squares(dense, mean, arithmetic='fp32')
    bf = centered_squares(dense, mean, arithmetic='original')
    assert fp.dtype == jnp.float32 and bf.dtype == jnp.bfloat16
    assert float(fp[0,0]) != float(bf[0,0])
