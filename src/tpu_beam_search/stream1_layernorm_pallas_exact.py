"""Correctness-first all-Pallas LayerNorm ResMLP inference.

Every arithmetic model operator from the prepacked embedding lookup through
the Q head is issued as a Pallas custom call.  JAX is used only to construct
and shard those calls and as an external reference in benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .sharding import make_sharded_inference
from .stream1_architecture import (
    DenseWeights,
    LayerNormDenseWeights,
    LayerNormResidualWeights,
    LayerNormStream1Weights,
    NormalizationKind,
    Stream1Architecture,
)
from .stream1_embedding_experimental import (
    BankedEmbedding,
    flat_embedding_prepacked,
)
from .stream1_layernorm_exact import prepare_exact_layernorm_inference_weights
from .stream1_layernorm_pallas import pallas_layernorm_dense
from .stream1_layernorm_reference import _validate_layernorm_shapes
from .tpu_layout import pad_to_multiple


_LAYER_NORM_ARITHMETICS = ("legacy_bf16", "hlo_mixed")


class PallasExactWeights(NamedTuple):
    """Runtime weights for the all-Pallas path."""

    embedding: BankedEmbedding
    input: LayerNormDenseWeights
    residuals: tuple[LayerNormResidualWeights, ...]
    output: DenseWeights


class PallasExactStage(NamedTuple):
    """One observable operator boundary in the diagnostic baseline."""

    name: str
    value: jax.Array


@dataclass(frozen=True)
class PallasExactConfig:
    """Independent tile and arithmetic choices for the three Dense families."""

    embedding_bm: int = 4096
    input_bm: int = 128
    input_bk: int = 128
    input_bn: int = 256
    residual_bm: int = 128
    residual_bk: int = 128
    residual_bn: int = 256
    head_bm: int = 256
    head_bk: int = 1024
    head_bn: int = 128
    dense_rounding: str = "late"
    layernorm_arithmetic: str = "hlo_mixed"

    def __post_init__(self) -> None:
        tiles = (
            self.embedding_bm,
            self.input_bm,
            self.input_bk,
            self.input_bn,
            self.residual_bm,
            self.residual_bk,
            self.residual_bn,
            self.head_bm,
            self.head_bk,
            self.head_bn,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in tiles):
            raise ValueError("all Pallas exact tiles must be positive integers")
        if self.dense_rounding not in ("late", "bf16_before_bias"):
            raise ValueError("dense_rounding must be 'late' or 'bf16_before_bias'")
        if self.layernorm_arithmetic not in _LAYER_NORM_ARITHMETICS:
            raise ValueError(
                f"layernorm_arithmetic must be one of {_LAYER_NORM_ARITHMETICS}"
            )


def prepare_pallas_exact_weights(
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
) -> PallasExactWeights:
    """Validate and prepack one BF16 checkpoint outside the inference call."""

    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)
    return PallasExactWeights(
        embedding=prepared.embedding,
        input=prepared.input,
        residuals=prepared.residuals,
        output=prepared.output,
    )


def _validate_prepared_weights(
    states: jax.Array,
    weights: PallasExactWeights,
    architecture: Stream1Architecture,
) -> None:
    if architecture.NORMALIZATION is not NormalizationKind.LAYER_NORM:
        raise ValueError("Pallas exact inference requires LayerNorm architecture")
    if states.ndim != 2 or states.shape[1] != architecture.STATE_STORAGE_LEN:
        raise ValueError(
            f"states shape must be [batch, {architecture.STATE_STORAGE_LEN}]"
        )
    if states.dtype != jnp.uint8:
        raise ValueError("states must use uint8 storage")
    if not isinstance(weights, PallasExactWeights):
        raise TypeError("weights must be PallasExactWeights")
    phases = architecture.EMBED_DIM // math.gcd(architecture.EMBED_DIM, 128)
    expected_banks = (phases, 128, 128)
    if (weights.embedding.low.shape != expected_banks
            or weights.embedding.high.shape != expected_banks):
        raise ValueError(f"embedding bank shapes must both be {expected_banks}")
    logical = LayerNormStream1Weights(
        embedding=jax.ShapeDtypeStruct(
            (architecture.NUM_CLASSES, architecture.EMBED_DIM), jnp.bfloat16,
        ),
        input=weights.input,
        residuals=weights.residuals,
        output=weights.output,
    )
    _validate_layernorm_shapes(states, logical, architecture)


def _layer_norm_activation_kernel(
    values_ref,
    scale_ref,
    bias_ref,
    skip_ref,
    output_ref,
    *,
    logical_width: int,
    epsilon: float,
    arithmetic: str,
    add_skip: bool,
    relu: bool,
):
    values_bf16 = values_ref[...].astype(jnp.bfloat16)
    columns = jax.lax.broadcasted_iota(jnp.int32, values_bf16.shape, 1)
    valid = columns < logical_width
    if arithmetic == "hlo_mixed":
        values = values_bf16.astype(jnp.float32)
        masked = jnp.where(valid, values, 0.0)
        mean = (jnp.sum(masked, axis=1, keepdims=True) / logical_width).astype(
            jnp.bfloat16
        )
        centered = jnp.where(valid, values - mean.astype(jnp.float32), 0.0)
        variance = (
            jnp.sum(centered * centered, axis=1, keepdims=True) / logical_width
        ).astype(jnp.bfloat16)
        eps = jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
        invstd = jax.lax.rsqrt(variance.astype(jnp.float32) + eps).astype(
            jnp.bfloat16
        )
        normalized = centered * invstd.astype(jnp.float32)
        result = (
            normalized * scale_ref[...].astype(jnp.float32)[None, :]
            + bias_ref[...].astype(jnp.float32)[None, :]
        ).astype(jnp.bfloat16)
    else:
        masked = jnp.where(valid, values_bf16, jnp.bfloat16(0))
        mean = jnp.sum(masked, axis=1, keepdims=True) / logical_width
        centered = jnp.where(valid, values_bf16 - mean, jnp.bfloat16(0))
        variance = jnp.sum(centered * centered, axis=1, keepdims=True) / logical_width
        result = (
            centered
            * jax.lax.rsqrt(variance + epsilon)
            * scale_ref[...][None, :]
            + bias_ref[...][None, :]
        ).astype(jnp.bfloat16)
    if add_skip:
        result = (result + skip_ref[...].astype(jnp.bfloat16)).astype(jnp.bfloat16)
    if relu:
        result = jnp.maximum(result, jnp.bfloat16(0)).astype(jnp.bfloat16)
    output_ref[...] = jnp.where(valid, result, jnp.bfloat16(0))


def pallas_exact_layer_norm_activation(
    values,
    scale,
    bias,
    *,
    skip=None,
    add_skip: bool = False,
    relu: bool = False,
    epsilon: float = 1e-5,
    bm: int = 128,
    width_alignment: int = 128,
    arithmetic: str = "hlo_mixed",
    interpret: bool = False,
):
    """LayerNorm plus optional residual and ReLU in one Pallas kernel."""

    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("values must be a nonempty matrix")
    rows, logical_width = values.shape
    if scale.shape != (logical_width,) or bias.shape != (logical_width,):
        raise ValueError("scale and bias must match values width")
    if add_skip and (skip is None or skip.shape != values.shape):
        raise ValueError("add_skip requires skip matching values shape")
    if arithmetic not in _LAYER_NORM_ARITHMETICS:
        raise ValueError(f"arithmetic must be one of {_LAYER_NORM_ARITHMETICS}")
    if not isinstance(bm, int) or bm <= 0 or not isinstance(width_alignment, int) \
            or width_alignment <= 0:
        raise ValueError("bm and width_alignment must be positive integers")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    padded_rows = pad_to_multiple(rows, bm)
    padded_width = pad_to_multiple(logical_width, width_alignment)
    padding = ((0, padded_rows - rows), (0, padded_width - logical_width))
    values_padded = jnp.pad(values.astype(jnp.bfloat16), padding)
    scale_padded = jnp.pad(
        scale.astype(jnp.bfloat16), (0, padded_width - logical_width)
    )
    bias_padded = jnp.pad(
        bias.astype(jnp.bfloat16), (0, padded_width - logical_width)
    )
    skip_source = values if skip is None else skip
    skip_padded = jnp.pad(skip_source.astype(jnp.bfloat16), padding)
    matrix_spec = pl.BlockSpec(
        (bm, padded_width), lambda row_block: (row_block, 0)
    )
    vector_spec = pl.BlockSpec((padded_width,), lambda row_block: (0,))
    call = pl.pallas_call(
        functools.partial(
            _layer_norm_activation_kernel,
            logical_width=logical_width,
            epsilon=epsilon,
            arithmetic=arithmetic,
            add_skip=add_skip,
            relu=relu,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[matrix_spec, vector_spec, vector_spec, matrix_spec],
            out_specs=matrix_spec,
            grid=(padded_rows // bm,),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_width), jnp.bfloat16
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=f"stream1_pallas_exact_ln_{arithmetic}",
    )
    return call(values_padded, scale_padded, bias_padded, skip_padded)[
        :rows, :logical_width
    ]


def _exact_dense(values, layer, *, bm, bk, bn, rounding, interpret):
    return pallas_layernorm_dense(
        values,
        layer.weight,
        layer.bias,
        bm=bm,
        bk=bk,
        bn=bn,
        dense_rounding=rounding,
        interpret=interpret,
    )


def pallas_exact_stage_names(
    architecture: Stream1Architecture,
) -> tuple[str, ...]:
    names = ["embedding", "input.dense", "input.layernorm_relu"]
    for index in range(architecture.RESIDUAL_COUNT):
        names.extend((
            f"residual.{index}.dense1",
            f"residual.{index}.layernorm1_relu",
            f"residual.{index}.dense2",
            f"residual.{index}.layernorm2_skip_relu",
        ))
    names.append("head.dense")
    return tuple(names)


def stream1_layernorm_pallas_exact_stages(
    states,
    weights: PallasExactWeights,
    architecture: Stream1Architecture,
    *,
    config: PallasExactConfig = PallasExactConfig(),
    interpret: bool = False,
) -> tuple[PallasExactStage, ...]:
    """Run the transparent 4*N+4 all-Pallas diagnostic baseline."""

    _validate_prepared_weights(states, weights, architecture)
    logical_states = states[:, : architecture.STATE_LEN]
    stages = []
    hidden = flat_embedding_prepacked(
        logical_states,
        weights.embedding,
        embed_dim=architecture.EMBED_DIM,
        bm=config.embedding_bm,
        interpret=interpret,
    )
    stages.append(PallasExactStage("embedding", hidden))
    hidden = _exact_dense(
        hidden,
        weights.input.dense,
        bm=config.input_bm,
        bk=config.input_bk,
        bn=config.input_bn,
        rounding=config.dense_rounding,
        interpret=interpret,
    )
    stages.append(PallasExactStage("input.dense", hidden))
    hidden = pallas_exact_layer_norm_activation(
        hidden,
        weights.input.normalization.scale,
        weights.input.normalization.bias,
        relu=True,
        epsilon=architecture.LAYER_NORM_EPSILON,
        bm=config.input_bm,
        arithmetic=config.layernorm_arithmetic,
        interpret=interpret,
    )
    stages.append(PallasExactStage("input.layernorm_relu", hidden))
    for index, block in enumerate(weights.residuals):
        skip = hidden
        branch = _exact_dense(
            hidden,
            block.first.dense,
            bm=config.residual_bm,
            bk=config.residual_bk,
            bn=config.residual_bn,
            rounding=config.dense_rounding,
            interpret=interpret,
        )
        stages.append(PallasExactStage(f"residual.{index}.dense1", branch))
        branch = pallas_exact_layer_norm_activation(
            branch,
            block.first.normalization.scale,
            block.first.normalization.bias,
            relu=True,
            epsilon=architecture.LAYER_NORM_EPSILON,
            bm=config.residual_bm,
            arithmetic=config.layernorm_arithmetic,
            interpret=interpret,
        )
        stages.append(PallasExactStage(
            f"residual.{index}.layernorm1_relu", branch,
        ))
        branch = _exact_dense(
            branch,
            block.second.dense,
            bm=config.residual_bm,
            bk=config.residual_bk,
            bn=config.residual_bn,
            rounding=config.dense_rounding,
            interpret=interpret,
        )
        stages.append(PallasExactStage(f"residual.{index}.dense2", branch))
        hidden = pallas_exact_layer_norm_activation(
            branch,
            block.second.normalization.scale,
            block.second.normalization.bias,
            skip=skip,
            add_skip=True,
            relu=True,
            epsilon=architecture.LAYER_NORM_EPSILON,
            bm=config.residual_bm,
            arithmetic=config.layernorm_arithmetic,
            interpret=interpret,
        )
        stages.append(PallasExactStage(
            f"residual.{index}.layernorm2_skip_relu", hidden,
        ))
    output = _exact_dense(
        hidden,
        weights.output,
        bm=config.head_bm,
        bk=config.head_bk,
        bn=config.head_bn,
        rounding=config.dense_rounding,
        interpret=interpret,
    )
    stages.append(PallasExactStage("head.dense", output))
    return tuple(stages)


def stream1_layernorm_pallas_exact_inference(
    states,
    weights: PallasExactWeights,
    architecture: Stream1Architecture,
    *,
    config: PallasExactConfig = PallasExactConfig(),
    interpret: bool = False,
):
    """Return only the final Q tensor from the all-Pallas diagnostic path."""

    return stream1_layernorm_pallas_exact_stages(
        states, weights, architecture, config=config, interpret=interpret,
    )[-1].value


def make_sharded_pallas_exact_inference(
    architecture: Stream1Architecture,
    *,
    mesh,
    weights_example: PallasExactWeights,
    config: PallasExactConfig = PallasExactConfig(),
    interpret: bool = False,
) -> Callable:
    """Compile independent state shards with replicated all-Pallas weights."""

    if "core" not in mesh.axis_names:
        raise ValueError("mesh must contain the 'core' axis")
    dummy = jax.ShapeDtypeStruct(
        (1, architecture.STATE_STORAGE_LEN), jnp.uint8,
    )
    _validate_prepared_weights(dummy, weights_example, architecture)
    return make_sharded_inference(
        lambda states, weights: stream1_layernorm_pallas_exact_inference(
            states,
            weights,
            architecture,
            config=config,
            interpret=interpret,
        ),
        mesh=mesh,
        weights_example=weights_example,
    )
