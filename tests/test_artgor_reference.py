from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tpu_beam_search.artgor_reference import artgor_reference_apply


def _params():
    return {
        "encoding": "embedding",
        "num_classes": 5,
        "embed": jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10,
        "input_stack": [
            {
                "lin_w": jnp.arange(48, dtype=jnp.float32).reshape(6, 8) / 50,
                "lin_b": jnp.linspace(-0.2, 0.2, 8),
                "ln_gamma": jnp.linspace(0.5, 1.5, 8),
                "ln_beta": jnp.linspace(-0.1, 0.1, 8),
            }
        ],
        "res_blocks": [
            {
                "lin1_w": jnp.eye(8, dtype=jnp.float32) * 0.5,
                "lin1_b": jnp.linspace(-0.1, 0.1, 8),
                "ln1_gamma": jnp.ones(8, dtype=jnp.float32),
                "ln1_beta": jnp.zeros(8, dtype=jnp.float32),
                "lin2_w": jnp.flip(jnp.eye(8, dtype=jnp.float32), axis=1) * 0.25,
                "lin2_b": jnp.linspace(0.1, -0.1, 8),
                "ln2_gamma": jnp.linspace(0.8, 1.2, 8),
                "ln2_beta": jnp.linspace(-0.05, 0.05, 8),
            }
        ],
        "head_w": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 20,
        "head_b": jnp.linspace(-0.3, 0.3, 4),
    }


def _literal_apply(params, states, dtype):
    embedded = params["embed"][states.astype(jnp.int32)]
    hidden = embedded.reshape(states.shape[0], -1).astype(dtype)

    def layer_norm(values, gamma, beta):
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
        return (values - mean) * jax_lax_rsqrt(variance + 1e-5) * gamma + beta

    layer = params["input_stack"][0]
    hidden = hidden @ layer["lin_w"].astype(dtype) + layer["lin_b"].astype(dtype)
    hidden = layer_norm(
        hidden,
        layer["ln_gamma"].astype(dtype),
        layer["ln_beta"].astype(dtype),
    )
    hidden = jnp.maximum(hidden, 0)
    for block in params["res_blocks"]:
        skip = hidden
        branch = hidden @ block["lin1_w"].astype(dtype) + block["lin1_b"].astype(dtype)
        branch = layer_norm(
            branch,
            block["ln1_gamma"].astype(dtype),
            block["ln1_beta"].astype(dtype),
        )
        branch = jnp.maximum(branch, 0)
        branch = branch @ block["lin2_w"].astype(dtype) + block["lin2_b"].astype(dtype)
        branch = layer_norm(
            branch,
            block["ln2_gamma"].astype(dtype),
            block["ln2_beta"].astype(dtype),
        )
        hidden = jnp.maximum(skip + branch, 0)
    return hidden @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)


def jax_lax_rsqrt(values):
    import jax

    return jax.lax.rsqrt(values)


def test_artgor_reference_matches_literal_operation_order():
    params = _params()
    states = jnp.asarray([[0, 1], [4, 3], [2, 2]], dtype=jnp.uint8)
    expected = _literal_apply(params, states, jnp.bfloat16)
    actual = artgor_reference_apply(params, states, dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_artgor_reference_is_finite_and_deterministic_on_edge_states():
    params = _params()
    states = jnp.asarray([[0, 0], [4, 4]], dtype=jnp.uint8)
    first = artgor_reference_apply(params, states, dtype=jnp.bfloat16)
    second = artgor_reference_apply(params, states, dtype=jnp.bfloat16)
    assert first.shape == (2, 4)
    assert bool(jnp.all(jnp.isfinite(first)))
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
