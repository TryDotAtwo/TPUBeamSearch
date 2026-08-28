from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .stream1_pallas import pallas_dense_linear, pallas_fused_folded_hidden
from .stream1_reference import folded_input_linear
from .stream1_architecture import (
    DenseWeights,
    ResidualWeights,
    Stream1Architecture,
    Stream1Weights,
)


def _validate_shapes(
    states: jax.Array,
    weights: Stream1Weights,
    architecture: Stream1Architecture,
) -> None:
    a = architecture
    if states.ndim != 2 or states.shape[1] != a.STATE_STORAGE_LEN:
        raise ValueError(
            f"states shape must be [batch, {a.STATE_STORAGE_LEN}]"
        )
    expected = (
        ("input weight", weights.input.weight.shape, (a.STATE_LEN * a.NUM_CLASSES, a.HIDDEN1)),
        ("input bias", weights.input.bias.shape, (a.HIDDEN1,)),
        ("hidden weight", weights.hidden.weight.shape, (a.HIDDEN1, a.HIDDEN2)),
        ("hidden bias", weights.hidden.bias.shape, (a.HIDDEN2,)),
        ("output weight", weights.output.weight.shape, (a.HIDDEN2, a.MOVE_COUNT)),
        ("output bias", weights.output.bias.shape, (a.MOVE_COUNT,)),
    )
    for name, actual, wanted in expected:
        if actual != wanted:
            raise ValueError(f"{name} shape must be {wanted}, got {actual}")
    if len(weights.residuals) != a.RESIDUAL_COUNT:
        raise ValueError(
            "residual weights count must equal RESIDUAL_COUNT: "
            f"{len(weights.residuals)} != {a.RESIDUAL_COUNT}"
        )
    for index, residual in enumerate(weights.residuals):
        for layer_name, layer in (("first", residual.first), ("second", residual.second)):
            if layer.weight.shape != (a.HIDDEN2, a.HIDDEN2):
                raise ValueError(
                    f"residual {index} {layer_name} weight shape must be "
                    f"{(a.HIDDEN2, a.HIDDEN2)}"
                )
            if layer.bias.shape != (a.HIDDEN2,):
                raise ValueError(
                    f"residual {index} {layer_name} bias shape must be {(a.HIDDEN2,)}"
                )


def _reference_dense(values, layer: DenseWeights, *, relu: bool):
    result = (
        values.astype(jnp.float32) @ layer.weight.astype(jnp.float32)
        + layer.bias.astype(jnp.float32)
    )
    if relu:
        result = jnp.maximum(result, 0.0)
    return result.astype(jnp.bfloat16)


def _host_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _fold_checkpoint_linear_bn(
    state_dict: Mapping[str, object],
    linear: str,
    batch_norm: str,
    *,
    BN_EPSILON: float,
    dtype,
) -> DenseWeights:
    weight_out_in = _host_array(state_dict[f"{linear}.weight"])
    bias = _host_array(state_dict[f"{linear}.bias"])
    gamma = _host_array(state_dict[f"{batch_norm}.weight"])
    beta = _host_array(state_dict[f"{batch_norm}.bias"])
    mean = _host_array(state_dict[f"{batch_norm}.running_mean"])
    variance = _host_array(state_dict[f"{batch_norm}.running_var"])
    scale = gamma / np.sqrt(variance + BN_EPSILON)
    folded_weight = (weight_out_in * scale[:, None]).T
    folded_bias = (bias - mean) * scale + beta
    return DenseWeights(
        weight=jnp.asarray(folded_weight, dtype=dtype),
        bias=jnp.asarray(folded_bias, dtype=dtype),
    )


def stream1_weights_from_pytorch_state_dict(
    state_dict: Mapping[str, object],
    architecture: Stream1Architecture,
    *,
    BN_EPSILON: float = 1e-5,
    dtype=jnp.bfloat16,
) -> Stream1Weights:
    """Convert an eval-mode Pilgrim state_dict into TPU inference weights."""

    if BN_EPSILON < 0:
        raise ValueError("BN_EPSILON must be non-negative")
    input_layer = _fold_checkpoint_linear_bn(
        state_dict,
        "input_layer",
        "bn1",
        BN_EPSILON=BN_EPSILON,
        dtype=dtype,
    )
    hidden_layer = _fold_checkpoint_linear_bn(
        state_dict,
        "hidden_layer",
        "bn2",
        BN_EPSILON=BN_EPSILON,
        dtype=dtype,
    )
    residuals = []
    for index in range(architecture.RESIDUAL_COUNT):
        prefix = f"residual_blocks.{index}"
        residuals.append(
            ResidualWeights(
                first=_fold_checkpoint_linear_bn(
                    state_dict,
                    f"{prefix}.fc1",
                    f"{prefix}.bn1",
                    BN_EPSILON=BN_EPSILON,
                    dtype=dtype,
                ),
                second=_fold_checkpoint_linear_bn(
                    state_dict,
                    f"{prefix}.fc2",
                    f"{prefix}.bn2",
                    BN_EPSILON=BN_EPSILON,
                    dtype=dtype,
                ),
            )
        )
    output = DenseWeights(
        weight=jnp.asarray(
            _host_array(state_dict["output_layer.weight"]).T, dtype=dtype
        ),
        bias=jnp.asarray(_host_array(state_dict["output_layer.bias"]), dtype=dtype),
    )
    weights = Stream1Weights(
        input=input_layer,
        hidden=hidden_layer,
        residuals=tuple(residuals),
        output=output,
    )
    _validate_shapes(
        jax.ShapeDtypeStruct(
            (1, architecture.STATE_STORAGE_LEN), jnp.uint8
        ),
        weights,
        architecture,
    )
    return weights


def stream1_reference_inference(
    states: jax.Array,
    weights: Stream1Weights,
    architecture: Stream1Architecture,
) -> jax.Array:
    """Inference with the same BF16 layer boundaries as the Pallas path."""

    _validate_shapes(states, weights, architecture)
    logical_states = states[:, : architecture.STATE_LEN]
    hidden = folded_input_linear(
        logical_states,
        weights.input.weight,
        weights.input.bias,
        NUM_CLASSES=architecture.NUM_CLASSES,
    )
    hidden = jnp.maximum(hidden, 0.0).astype(jnp.bfloat16)
    hidden = _reference_dense(hidden, weights.hidden, relu=True)
    for residual in weights.residuals:
        skip = hidden
        branch = _reference_dense(hidden, residual.first, relu=True)
        branch = _reference_dense(branch, residual.second, relu=False)
        hidden = jnp.maximum(
            skip.astype(jnp.float32) + branch.astype(jnp.float32), 0.0
        ).astype(jnp.bfloat16)
    return _reference_dense(hidden, weights.output, relu=False)


def stream1_pallas_inference(
    states: jax.Array,
    weights: Stream1Weights,
    architecture: Stream1Architecture,
    *,
    bm: int = 256,
    bk_input: int = 128,
    bn_input: int = 512,
    bk_hidden: int = 256,
    bn_hidden: int = 512,
    bk_residual: int = 256,
    bn_residual: int = 512,
    bk_output: int = 512,
    bn_output: int = 256,
    interpret: bool = False,
) -> jax.Array:
    """Complete MLP inference; architecture and tile arguments are compile-time static."""

    _validate_shapes(states, weights, architecture)
    hidden = pallas_fused_folded_hidden(
        states,
        weights.input.weight,
        weights.input.bias,
        weights.hidden.weight,
        weights.hidden.bias,
        STATE_LEN=architecture.STATE_LEN,
        NUM_CLASSES=architecture.NUM_CLASSES,
        bm=bm,
        bk_input=bk_input,
        bn_input=bn_input,
        bk_hidden=bk_hidden,
        bn_hidden=bn_hidden,
        interpret=interpret,
    )
    for residual in weights.residuals:
        skip = hidden
        branch = pallas_dense_linear(
            hidden,
            residual.first.weight,
            residual.first.bias,
            bm=bm,
            bk=bk_residual,
            bn=bn_residual,
            relu=True,
            interpret=interpret,
        )
        branch = pallas_dense_linear(
            branch,
            residual.second.weight,
            residual.second.bias,
            bm=bm,
            bk=bk_residual,
            bn=bn_residual,
            relu=False,
            interpret=interpret,
        )
        hidden = jnp.maximum(
            skip.astype(jnp.float32) + branch.astype(jnp.float32), 0.0
        ).astype(jnp.bfloat16)
    return pallas_dense_linear(
        hidden,
        weights.output.weight,
        weights.output.bias,
        bm=bm,
        bk=bk_output,
        bn=bn_output,
        relu=False,
        interpret=interpret,
    )


def make_jitted_stream1_inference(
    architecture: Stream1Architecture,
    *,
    backend: str = "pallas",
    **pallas_options,
):
    """Compile one executable per architecture, input shape/dtype, and tile plan."""

    if backend == "reference":
        return jax.jit(
            lambda states, weights: stream1_reference_inference(
                states, weights, architecture
            )
        )
    if backend == "pallas":
        return jax.jit(
            lambda states, weights: stream1_pallas_inference(
                states,
                weights,
                architecture,
                **pallas_options,
            )
        )
    raise ValueError("backend must be 'pallas' or 'reference'")
