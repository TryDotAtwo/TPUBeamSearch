"""Stable exact inference API for Artgor's Cube555 LayerNorm Q model.

The selected TPU path is deliberately two separately compiled calls.  Calling
``ArtgorExactInference`` from Python preserves that boundary; enclosing the
composition in another ``jax.jit`` is outside the validated contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from .sharding import make_sharded_inference
from .stream1_architecture import (
    LayerNormStream1Weights,
    Stream1Architecture,
)
from .stream1_layernorm_exact import (
    ExactLayerNormInferenceWeights,
    prepare_exact_layernorm_inference_weights,
    stream1_layernorm_exact_prefix,
)
from .stream1_layernorm_pallas import pallas_layernorm_dense
from .stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)


ARTGOR_Q_CONTRACT = (150, 150, 24, 1024, 1024, 10, 30)


@dataclass(frozen=True)
class ArtgorExactConfig:
    """Frozen production tiling plus smaller interpreter-test overrides."""

    prefix_bm: int = 4096
    head_bm: int = 256
    head_bk: int = 1024
    head_bn: int = 128
    dense_rounding: str = "late"
    inference_chunk: int = 32768
    parent_chunk: int = 131072

    def validate(self) -> None:
        integer_fields = (
            self.prefix_bm,
            self.head_bm,
            self.head_bk,
            self.head_bn,
            self.inference_chunk,
            self.parent_chunk,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in integer_fields
        ):
            raise ValueError("all exact-inference sizes must be positive integers")
        if self.parent_chunk % self.inference_chunk:
            raise ValueError(
                "parent_chunk must divide into whole inference chunks"
            )
        if self.dense_rounding != "late":
            raise ValueError(
                "the published exact Artgor head requires late BF16 rounding"
            )


@dataclass(frozen=True)
class EngineDecision:
    requested: str
    selected: str
    reason: str


@dataclass(frozen=True)
class ArtgorExactInference:
    """Two separately compiled device-resident inference stages."""

    prefix: Callable
    head: Callable

    def __call__(self, states, weights):
        hidden = self.prefix(states, weights)
        return self.head(hidden, weights)


def choose_artgor_inference_engine(
    requested: str,
    blend_checkpoints: Sequence[str] | None,
    qv_consistency: float,
) -> EngineDecision:
    """Choose the exact engine or an explicit original-JAX fallback."""

    if requested not in ("exact_split", "original_jax"):
        raise ValueError(
            "INFERENCE_ENGINE must be 'exact_split' or 'original_jax'"
        )
    qv = float(qv_consistency)
    if not math.isfinite(qv):
        raise ValueError("QV_CONSISTENCY must be finite")
    if requested == "original_jax":
        return EngineDecision(
            requested=requested,
            selected="original_jax",
            reason="original_jax was requested explicitly",
        )
    if blend_checkpoints:
        return EngineDecision(
            requested=requested,
            selected="original_jax",
            reason=(
                "BLEND_CHECKPOINTS is not covered by the exact split gate; "
                "using original_jax"
            ),
        )
    if qv != 0.0:
        return EngineDecision(
            requested=requested,
            selected="original_jax",
            reason=(
                "nonzero QV_CONSISTENCY needs the auxiliary value head; "
                "using original_jax"
            ),
        )
    return EngineDecision(
        requested=requested,
        selected="exact_split",
        reason="single-checkpoint Q-only configuration is exact-split eligible",
    )


def prepare_artgor_exact_inference_from_weights(
    weights: LayerNormStream1Weights,
    architecture: Stream1Architecture,
    *,
    mesh,
    config: ArtgorExactConfig = ArtgorExactConfig(),
    interpret: bool = False,
) -> tuple[ArtgorExactInference, ExactLayerNormInferenceWeights]:
    """Prepare weights and compile the selected prefix and Pallas head."""

    config.validate()
    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)

    def prefix_local(states, runtime_weights):
        return stream1_layernorm_exact_prefix(
            states,
            runtime_weights,
            architecture,
            bm=config.prefix_bm,
            interpret=interpret,
        )

    def head_local(hidden, runtime_weights):
        return pallas_layernorm_dense(
            hidden,
            runtime_weights.output.weight,
            runtime_weights.output.bias,
            bm=config.head_bm,
            bk=config.head_bk,
            bn=config.head_bn,
            dense_rounding=config.dense_rounding,
            interpret=interpret,
        )

    engine = ArtgorExactInference(
        prefix=make_sharded_inference(
            prefix_local, mesh=mesh, weights_example=prepared
        ),
        head=make_sharded_inference(
            head_local, mesh=mesh, weights_example=prepared
        ),
    )
    return engine, prepared


def prepare_artgor_exact_inference(
    params: Mapping[str, Any],
    *,
    mesh,
    config: ArtgorExactConfig = ArtgorExactConfig(),
    state_storage_len: int = 150,
    interpret: bool = False,
) -> tuple[
    ArtgorExactInference,
    ExactLayerNormInferenceWeights,
    Stream1Architecture,
]:
    """Convert one Artgor checkpoint and build the validated exact engine."""

    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=state_storage_len
    )
    contract = (
        architecture.STATE_LEN,
        architecture.NUM_CLASSES,
        architecture.EMBED_DIM,
        architecture.HIDDEN1,
        architecture.HIDDEN2,
        architecture.RESIDUAL_COUNT,
        architecture.MOVE_COUNT,
    )
    if contract != ARTGOR_Q_CONTRACT:
        raise ValueError(
            f"checkpoint contract {contract} is not the published "
            f"Artgor Q contract {ARTGOR_Q_CONTRACT}"
        )
    weights = layernorm_stream1_weights_from_artgor_params(
        params, architecture
    )
    engine, prepared = prepare_artgor_exact_inference_from_weights(
        weights,
        architecture,
        mesh=mesh,
        config=config,
        interpret=interpret,
    )
    return engine, prepared, architecture
