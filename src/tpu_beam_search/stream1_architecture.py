from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, NamedTuple

import jax


def _shape(value) -> tuple[int, ...]:
    return tuple(value.shape)


class NormalizationKind(str, Enum):
    FOLDED_BATCH_NORM = "folded_batch_norm"
    LAYER_NORM = "layer_norm"


class InputEncodingKind(str, Enum):
    EMBEDDING_GATHER = "embedding_gather"
    VIRTUAL_ONE_HOT_MXU = "virtual_one_hot_mxu"
    FUSED_VIRTUAL_ONE_HOT = "fused_virtual_one_hot"


class DenseWeights(NamedTuple):
    """A folded affine layer with weights stored as [input, output]."""

    weight: jax.Array
    bias: jax.Array


class ResidualWeights(NamedTuple):
    first: DenseWeights
    second: DenseWeights


class Stream1Weights(NamedTuple):
    input: DenseWeights
    hidden: DenseWeights
    residuals: tuple[ResidualWeights, ...]
    output: DenseWeights


class LayerNormWeights(NamedTuple):
    scale: jax.Array
    bias: jax.Array


class LayerNormDenseWeights(NamedTuple):
    dense: DenseWeights
    normalization: LayerNormWeights


class LayerNormResidualWeights(NamedTuple):
    first: LayerNormDenseWeights
    second: LayerNormDenseWeights


class LayerNormStream1Weights(NamedTuple):
    embedding: jax.Array
    input: LayerNormDenseWeights
    residuals: tuple[LayerNormResidualWeights, ...]
    output: DenseWeights
    fused_input_weight: jax.Array | None = None


@dataclass(frozen=True)
class Stream1Architecture:
    """Compile-time logical model shape, independent of Pallas tile padding."""

    STATE_LEN: int
    STATE_STORAGE_LEN: int
    NUM_CLASSES: int
    HIDDEN1: int
    HIDDEN2: int
    RESIDUAL_COUNT: int
    MOVE_COUNT: int
    NORMALIZATION: NormalizationKind = NormalizationKind.FOLDED_BATCH_NORM
    INPUT_ENCODING: InputEncodingKind = InputEncodingKind.VIRTUAL_ONE_HOT_MXU
    EMBED_DIM: int = 0
    LAYER_NORM_EPSILON: float = 1e-5

    def __post_init__(self) -> None:
        positive = (
            "STATE_LEN",
            "STATE_STORAGE_LEN",
            "NUM_CLASSES",
            "HIDDEN1",
            "HIDDEN2",
            "MOVE_COUNT",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.RESIDUAL_COUNT < 0:
            raise ValueError("RESIDUAL_COUNT must be non-negative")
        if self.STATE_STORAGE_LEN < self.STATE_LEN:
            raise ValueError("STATE_STORAGE_LEN must be at least STATE_LEN")
        if self.EMBED_DIM < 0:
            raise ValueError("EMBED_DIM must be non-negative")
        if self.LAYER_NORM_EPSILON < 0:
            raise ValueError("LAYER_NORM_EPSILON must be non-negative")
        if self.NORMALIZATION is NormalizationKind.LAYER_NORM and self.EMBED_DIM <= 0:
            raise ValueError("LayerNorm architecture requires a positive EMBED_DIM")

    @classmethod
    def from_pytorch_state_dict(
        cls,
        state_dict: Mapping[str, object],
        *,
        STATE_LEN: int,
        STATE_STORAGE_LEN: int,
        NUM_CLASSES: int,
    ) -> "Stream1Architecture":
        input_shape = _shape(state_dict["input_layer.weight"])
        hidden_shape = _shape(state_dict["hidden_layer.weight"])
        output_shape = _shape(state_dict["output_layer.weight"])
        if len(input_shape) != 2 or input_shape[1] != STATE_LEN * NUM_CLASSES:
            raise ValueError("checkpoint input width must equal STATE_LEN * NUM_CLASSES")
        HIDDEN1 = input_shape[0]
        if len(hidden_shape) != 2 or hidden_shape[1] != HIDDEN1:
            raise ValueError("checkpoint hidden layer does not follow input layer")
        HIDDEN2 = hidden_shape[0]
        if len(output_shape) != 2 or output_shape[1] != HIDDEN2:
            raise ValueError("checkpoint output layer does not follow hidden layer")
        residual_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("residual_blocks.")
                and key.endswith(".fc1.weight")
            }
        )
        if residual_indices != list(range(len(residual_indices))):
            raise ValueError("checkpoint residual block indices must be contiguous")
        for index in residual_indices:
            for layer in ("fc1", "fc2"):
                shape = _shape(
                    state_dict[f"residual_blocks.{index}.{layer}.weight"]
                )
                if shape != (HIDDEN2, HIDDEN2):
                    raise ValueError("checkpoint residual weights must be square HIDDEN2")
        return cls(
            STATE_LEN=STATE_LEN,
            STATE_STORAGE_LEN=STATE_STORAGE_LEN,
            NUM_CLASSES=NUM_CLASSES,
            HIDDEN1=HIDDEN1,
            HIDDEN2=HIDDEN2,
            RESIDUAL_COUNT=len(residual_indices),
            MOVE_COUNT=output_shape[0],
        )

    @classmethod
    def from_artgor_params(
        cls,
        params: Mapping[str, object],
        *,
        STATE_STORAGE_LEN: int,
    ) -> "Stream1Architecture":
        if params.get("encoding", "embedding") != "embedding":
            raise ValueError("Artgor LayerNorm architecture requires embedding encoding")
        STATE_LEN = int(params["state_size"])
        NUM_CLASSES = int(params["num_classes"])
        embedding_shape = _shape(params["embed"])
        if len(embedding_shape) != 2 or embedding_shape[0] != NUM_CLASSES:
            raise ValueError(
                "embedding shape must be [NUM_CLASSES, EMBED_DIM]"
            )
        EMBED_DIM = embedding_shape[1]
        input_stack = params["input_stack"]
        if len(input_stack) != 1:
            raise ValueError("input_stack must contain exactly one Linear/LayerNorm stage")
        input_layer = input_stack[0]
        input_weight_shape = _shape(input_layer["lin_w"])
        if (
            len(input_weight_shape) != 2
            or input_weight_shape[0] != STATE_LEN * EMBED_DIM
        ):
            raise ValueError(
                "input_stack weight width must equal STATE_LEN * EMBED_DIM"
            )
        HIDDEN = input_weight_shape[1]
        expected_vector = (HIDDEN,)
        for name in ("lin_b", "ln_gamma", "ln_beta"):
            if _shape(input_layer[name]) != expected_vector:
                raise ValueError(f"input_stack {name} shape must be {expected_vector}")

        residuals = params["res_blocks"]
        for index, residual in enumerate(residuals):
            for name in ("lin1_w", "lin2_w"):
                if _shape(residual[name]) != (HIDDEN, HIDDEN):
                    raise ValueError(
                        f"residual {index} {name} shape must be {(HIDDEN, HIDDEN)}"
                    )
            for name in (
                "lin1_b",
                "ln1_gamma",
                "ln1_beta",
                "lin2_b",
                "ln2_gamma",
                "ln2_beta",
            ):
                if _shape(residual[name]) != expected_vector:
                    raise ValueError(
                        f"residual {index} {name} shape must be {expected_vector}"
                    )

        head_shape = _shape(params["head_w"])
        if len(head_shape) != 2 or head_shape[0] != HIDDEN:
            raise ValueError("head weight input width must equal HIDDEN")
        MOVE_COUNT = head_shape[1]
        if _shape(params["head_b"]) != (MOVE_COUNT,):
            raise ValueError("head bias shape must equal MOVE_COUNT")
        if int(params.get("d_model", HIDDEN)) != HIDDEN:
            raise ValueError("d_model metadata must match input_stack output width")
        if int(params.get("output_dim", MOVE_COUNT)) != MOVE_COUNT:
            raise ValueError("output_dim metadata must match head width")

        return cls(
            STATE_LEN=STATE_LEN,
            STATE_STORAGE_LEN=STATE_STORAGE_LEN,
            NUM_CLASSES=NUM_CLASSES,
            HIDDEN1=HIDDEN,
            HIDDEN2=HIDDEN,
            RESIDUAL_COUNT=len(residuals),
            MOVE_COUNT=MOVE_COUNT,
            NORMALIZATION=NormalizationKind.LAYER_NORM,
            INPUT_ENCODING=InputEncodingKind.EMBEDDING_GATHER,
            EMBED_DIM=EMBED_DIM,
        )
