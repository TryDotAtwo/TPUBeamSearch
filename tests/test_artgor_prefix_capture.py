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


def test_capture_invstd_slot_has_unit_variance_value():
    from benchmarks.artgor_prefix_capture import captured_prefix
    _, states, architecture, weights = model_fixture()
    layer = weights.input._replace(dense=weights.input.dense._replace(
        weight=jnp.zeros_like(weights.input.dense.weight),
        bias=jnp.asarray([-1,1,-1,1,-1,1,-1,1],jnp.bfloat16)))
    result = captured_prefix(states,weights._replace(input=layer),architecture,include_invstd=True)
    assert result.shape == (2,4,8)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result[:,3],np.ones((2,8)))


def test_external_invstd_affine_uses_supplied_value_and_relu():
    from benchmarks.artgor_invstd_capture import external_invstd_affine, invstd_buffer
    dense=jnp.tile(jnp.asarray([-1,1],jnp.bfloat16),(128,64))
    mean=jnp.zeros_like(dense)
    invstd=invstd_buffer(dense,mean,interpret=True)
    np.testing.assert_array_equal(invstd,np.ones((128,128)))
    output=external_invstd_affine(dense,mean,invstd*2,
        jnp.ones((128,),jnp.bfloat16),jnp.zeros((128,),jnp.bfloat16),interpret=True)
    np.testing.assert_array_equal(output,np.tile([0,2],(128,64)))
