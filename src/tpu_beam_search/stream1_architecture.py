from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax


def _shape(value) -> tuple[int, ...]:
    return tuple(value.shape)


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
