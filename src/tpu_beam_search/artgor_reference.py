from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp


def _layer_norm(
    values: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    *,
    epsilon: float = 1e-5,
) -> jax.Array:
    """Match the operation order in Artgor's PyTorch-compatible JAX port."""

    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (
        (values - mean)
        * jax.lax.rsqrt(variance + epsilon)
        * gamma
        + beta
    )


def artgor_reference_apply(
    params: Mapping[str, Any],
    states: jax.Array,
    *,
    dtype=jnp.float32,
) -> jax.Array:
    """Exact single-model Q inference semantics from Artgor's `jax_model.apply`."""

    if params.get("encoding", "embedding") == "onehot":
        encoded = jax.nn.one_hot(
            states.astype(jnp.int32),
            params["num_classes"],
            dtype=dtype,
        )
        hidden = encoded.reshape(encoded.shape[0], -1)
    else:
        embedded = params["embed"][states.astype(jnp.int32)]
        hidden = embedded.reshape(embedded.shape[0], -1).astype(dtype)

    for layer in params["input_stack"]:
        hidden = (
            hidden @ layer["lin_w"].astype(dtype)
            + layer["lin_b"].astype(dtype)
        )
        hidden = _layer_norm(
            hidden,
            layer["ln_gamma"].astype(dtype),
            layer["ln_beta"].astype(dtype),
        )
        hidden = jax.nn.relu(hidden)

    for block in params["res_blocks"]:
        skip = hidden
        branch = (
            hidden @ block["lin1_w"].astype(dtype)
            + block["lin1_b"].astype(dtype)
        )
        branch = _layer_norm(
            branch,
            block["ln1_gamma"].astype(dtype),
            block["ln1_beta"].astype(dtype),
        )
        branch = jax.nn.relu(branch)
        branch = (
            branch @ block["lin2_w"].astype(dtype)
            + block["lin2_b"].astype(dtype)
        )
        branch = _layer_norm(
            branch,
            block["ln2_gamma"].astype(dtype),
            block["ln2_beta"].astype(dtype),
        )
        hidden = jax.nn.relu(skip + branch)

    output = (
        hidden @ params["head_w"].astype(dtype)
        + params["head_b"].astype(dtype)
    )
    if output.shape[-1] == 1:
        output = jnp.squeeze(output, axis=-1)
    return output
