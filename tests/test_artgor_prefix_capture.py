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


def test_capture_variance_preserves_bf16_slots():
    from benchmarks.artgor_prefix_capture import captured_prefix
    _, states, architecture, weights = model_fixture()
    layer=weights.input._replace(dense=weights.input.dense._replace(
        weight=jnp.zeros_like(weights.input.dense.weight),
        bias=jnp.asarray([-2,2,-2,2,-2,2,-2,2],jnp.bfloat16)))
    result=captured_prefix(states,weights._replace(input=layer),architecture,
                           include_invstd=True,include_variance=True)
    assert result.shape==(2,5,8) and result.dtype==jnp.bfloat16
    np.testing.assert_array_equal(result[:,3],np.full((2,8),0.5))
    np.testing.assert_array_equal(result[:,4],np.full((2,8),4))


def test_variance_pair_and_replay_preserve_separate_dtypes():
    from benchmarks.artgor_invstd_capture import variance_pair, variance_rsqrt
    dense=jnp.tile(jnp.asarray([-2,2],jnp.bfloat16),(128,64))
    variance,invstd=variance_pair(dense,jnp.zeros_like(dense),interpret=True)
    assert variance.dtype==jnp.float32 and invstd.dtype==jnp.bfloat16
    np.testing.assert_array_equal(variance,np.full((128,128),4))
    np.testing.assert_array_equal(invstd,np.full((128,128),0.5))
    for dtype in (jnp.bfloat16,jnp.float32):
        replay=variance_rsqrt(variance.astype(dtype),interpret=True)
        np.testing.assert_array_equal(replay,np.full((128,128),0.5))


def test_pair_chunks_preserve_device_major_order_and_dtypes():
    from benchmarks.artgor_invstd_capture import chunked_pair_host
    states=np.arange(16).reshape(8,2)
    first,second=chunked_pair_host(states,lambda x:(x.astype(np.float32),x.astype(np.int16)),devices=2,chunk_rows=2)
    np.testing.assert_array_equal(first,states)
    np.testing.assert_array_equal(second,states)
    assert first.dtype==np.float32 and second.dtype==np.int16
