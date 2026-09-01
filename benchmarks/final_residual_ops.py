"""Final-residual boundaries used to restore the canonical TPU Dense schedule."""
from __future__ import annotations

import jax

from benchmarks.execution_boundary_ops import candidate_encode
from tpu_beam_search.stream1_layernorm_reference import layer_norm_reference


FINAL_CUTS = (
    "before_final_block",
    "before_final_dense2",
    "after_final_dense2",
    "after_final_block",
)
FINAL_TAPS = (
    "before_final_dense2",
    "after_final_dense2",
    "after_final_block",
)
FINAL_BARRIERS = (
    "none",
    "before_final_block",
    "before_final_dense2",
    "after_final_dense2",
    "after_final_block",
    "before_and_after_final_dense2",
)


def _validate(config, architecture):
    if architecture.RESIDUAL_COUNT < 1:
        raise ValueError("final-residual experiment requires at least one block")
    if config.get("dense", "jax") != "jax" or config.get("norm", "jax") != "jax":
        raise ValueError("final-residual experiment preserves the JAX Dense/LN tail")
    if config.get("input_boundary", "none") != "none":
        raise ValueError("input boundaries are outside the final-residual experiment")
    barrier = config.get("final_barrier", "none")
    if barrier not in FINAL_BARRIERS:
        raise ValueError(f"unknown final-residual barrier: {barrier}")


def _normalized_dense(values, layer, *, epsilon, relu):
    dense = values @ layer.dense.weight + layer.dense.bias
    normalized = layer_norm_reference(
        dense, layer.normalization, epsilon=epsilon,
    )
    return jax.nn.relu(normalized) if relu else normalized


def _input_hidden(encoded, weights, *, epsilon):
    return _normalized_dense(encoded, weights.input, epsilon=epsilon, relu=True)


def _residual_block(hidden, block, *, epsilon):
    skip = hidden
    branch = _normalized_dense(hidden, block.first, epsilon=epsilon, relu=True)
    branch = _normalized_dense(branch, block.second, epsilon=epsilon, relu=False)
    return jax.nn.relu(skip + branch)


def _before_final_block(encoded, weights, *, epsilon):
    hidden = _input_hidden(encoded, weights, epsilon=epsilon)
    for block in weights.residuals[:-1]:
        hidden = _residual_block(hidden, block, epsilon=epsilon)
    return hidden


def _final_block(hidden, block, *, epsilon, barrier):
    if barrier == "before_final_block":
        hidden = jax.lax.optimization_barrier(hidden)
    skip = hidden
    branch = _normalized_dense(hidden, block.first, epsilon=epsilon, relu=True)
    before_dense2 = branch
    if barrier in ("before_final_dense2", "before_and_after_final_dense2"):
        branch = jax.lax.optimization_barrier(branch)
    dense2 = branch @ block.second.dense.weight + block.second.dense.bias
    if barrier in ("after_final_dense2", "before_and_after_final_dense2"):
        dense2 = jax.lax.optimization_barrier(dense2)
    branch = layer_norm_reference(
        dense2, block.second.normalization, epsilon=epsilon,
    )
    hidden = jax.nn.relu(skip + branch)
    if barrier == "after_final_block":
        hidden = jax.lax.optimization_barrier(hidden)
    return hidden, dict(
        before_final_dense2=before_dense2,
        after_final_dense2=dense2,
        after_final_block=hidden,
    )


def candidate_final_full(config, architecture, *, tap=None, interpret=False):
    """Build a monolith with one targeted final-block barrier or output tap."""
    _validate(config, architecture)
    if tap is not None and tap not in FINAL_TAPS:
        raise ValueError(f"unknown final-residual tap: {tap}")
    encode = candidate_encode(config, architecture, interpret=interpret)
    epsilon = architecture.LAYER_NORM_EPSILON
    barrier = config.get("final_barrier", "none")

    def call(states, weights):
        encoded = encode(states, weights)
        hidden = _before_final_block(encoded, weights, epsilon=epsilon)
        hidden, taps = _final_block(
            hidden, weights.residuals[-1], epsilon=epsilon, barrier=barrier,
        )
        q = hidden @ weights.output.weight + weights.output.bias
        return q if tap is None else (q, taps[tap])

    return call


def candidate_final_partition(config, architecture, *, cut, interpret=False):
    """Split the model at one final-block boundary without changing formulas."""
    _validate(config, architecture)
    if cut not in FINAL_CUTS:
        raise ValueError(f"unknown final-residual cut: {cut}")
    if config.get("final_barrier", "none") != "none":
        raise ValueError("a materialized cut and optimization barrier cannot be mixed")
    encode = candidate_encode(config, architecture, interpret=interpret)
    epsilon = architecture.LAYER_NORM_EPSILON

    def prefix(states, weights):
        encoded = encode(states, weights)
        skip = _before_final_block(encoded, weights, epsilon=epsilon)
        if cut == "before_final_block":
            return skip
        block = weights.residuals[-1]
        branch = _normalized_dense(
            skip, block.first, epsilon=epsilon, relu=True,
        )
        if cut == "before_final_dense2":
            return skip, branch
        dense2 = branch @ block.second.dense.weight + block.second.dense.bias
        if cut == "after_final_dense2":
            return skip, dense2
        branch = layer_norm_reference(
            dense2, block.second.normalization, epsilon=epsilon,
        )
        return jax.nn.relu(skip + branch)

    def suffix(intermediate, weights):
        block = weights.residuals[-1]
        if cut == "before_final_block":
            hidden = _residual_block(intermediate, block, epsilon=epsilon)
        elif cut == "before_final_dense2":
            skip, branch = intermediate
            dense2 = branch @ block.second.dense.weight + block.second.dense.bias
            branch = layer_norm_reference(
                dense2, block.second.normalization, epsilon=epsilon,
            )
            hidden = jax.nn.relu(skip + branch)
        elif cut == "after_final_dense2":
            skip, dense2 = intermediate
            branch = layer_norm_reference(
                dense2, block.second.normalization, epsilon=epsilon,
            )
            hidden = jax.nn.relu(skip + branch)
        else:
            hidden = intermediate
        return hidden @ weights.output.weight + weights.output.bias

    return prefix, suffix
