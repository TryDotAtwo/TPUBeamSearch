import jax.numpy as jnp
import numpy as np
from test_layernorm_followup import model_fixture


def test_capture_returns_dense_mean_and_output_in_bf16():
    import benchmarks.artgor_prefix_capture as capture
    _, states, architecture, weights = model_fixture()
    layer = weights.input._replace(
        dense=weights.input.dense._replace(weight=jnp.zeros_like(weights.input.dense.weight),
                                           bias=jnp.ones_like(weights.input.dense.bias)),
        normalization=weights.input.normalization._replace(bias=jnp.zeros_like(weights.input.normalization.bias)),
    )
    result = capture.captured_prefix(states, weights._replace(input=layer), architecture)
    assert result.shape == (2,3,8)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result[:,0],np.ones((2,8)))
    np.testing.assert_array_equal(result[:,1],np.ones((2,8)))
    np.testing.assert_array_equal(result[:,2],np.zeros((2,8)))
