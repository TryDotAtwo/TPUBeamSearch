import numpy as np
import jax.numpy as jnp
import pytest


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
