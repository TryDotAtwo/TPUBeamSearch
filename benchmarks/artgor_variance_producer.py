"""JAX producer probes on externally fixed Dense and mean buffers."""
import jax
import jax.numpy as jnp


def centered_squares(dense, mean, *, arithmetic):
    if arithmetic == 'fp32':
        dense, mean = dense.astype(jnp.float32), mean.astype(jnp.float32)
    elif arithmetic != 'original':
        raise ValueError('unknown producer arithmetic')
    return jnp.square(dense - mean)


def reduce_invstd(squares, *, arithmetic, epsilon=1e-5):
    if arithmetic == 'fp32':
        squares = squares.astype(jnp.float32)
        epsilon = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
    elif arithmetic != 'original':
        raise ValueError('unknown producer arithmetic')
    variance = jnp.mean(squares, axis=-1, keepdims=True)
    return jax.lax.rsqrt(variance + epsilon).astype(jnp.bfloat16)


def fused_invstd(dense, mean, *, arithmetic, epsilon=1e-5):
    return reduce_invstd(centered_squares(dense, mean, arithmetic=arithmetic),
                         arithmetic=arithmetic, epsilon=epsilon)
