"""JAX producer probes on externally fixed Dense and mean buffers."""
import jax
import jax.numpy as jnp
import numpy as np


def collect_separate(dense, mean, operation, *, devices=8, chunk_rows=256):
    from benchmarks.artgor_prefix_shape import chunked_host
    if len(dense) != len(mean):
        raise ValueError('Dense/mean row counts differ')
    return chunked_host(np.arange(len(mean)),
        lambda indices: operation(dense[indices], mean[indices]),
        devices=devices, chunk_rows=chunk_rows)


def separate_invstd(dense, mean, *, transposed=False, arithmetic='fp32', epsilon=1e-5):
    if mean.ndim != 1 or dense.ndim != 2 or dense.shape[1 if transposed else 0] != mean.shape[0]:
        raise ValueError('expected matrix and matching scalar mean vector')
    broadcast_mean = mean[None, :] if transposed else mean[:, None]
    squares = centered_squares(dense, broadcast_mean, arithmetic=arithmetic)
    if arithmetic == 'fp32':
        epsilon = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
    variance = jnp.mean(squares, axis=0 if transposed else 1)
    return jax.lax.rsqrt(variance + epsilon).astype(jnp.bfloat16)


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
