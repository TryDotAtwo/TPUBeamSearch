from __future__ import annotations

import jax.numpy as jnp


def apply_all_moves(parents, generators):
    """Return flattened candidates in parent-major, move-minor order."""
    children = jnp.take_along_axis(
        parents[:, None, :],
        generators[None, :, :],
        axis=2,
    )
    return children.reshape((-1, parents.shape[-1]))


def folded_input_linear(states, weight, bias, *, NUM_CLASSES: int):
    """Evaluate one-hot input linear without materializing the one-hot rows."""
    positions = jnp.arange(states.shape[1], dtype=jnp.int32)
    rows = positions[None, :] * NUM_CLASSES + states.astype(jnp.int32)
    selected = weight[rows]
    return selected.astype(jnp.float32).sum(axis=1) + bias.astype(jnp.float32)


def quantize_score(scores, *, SCORE_SCALE: int, SCORE_MAX_Q: float):
    clipped = jnp.clip(scores, 0.0, SCORE_MAX_Q)
    return jnp.rint(clipped * SCORE_SCALE).astype(jnp.uint32)
