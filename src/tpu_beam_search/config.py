from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeamConfig:
    MOVE_COUNT: int
    STATE_LEN: int
    STATE_STORAGE_LEN: int
    NUM_CLASSES: int
    SCORE_SCALE: int = 1024
    SCORE_MAX_Q: float = 300.0
    STATE_ALIGNMENT: int = 16

    @property
    def SCORE_MAX_KEY(self) -> int:
        return int(self.SCORE_MAX_Q) * self.SCORE_SCALE

    @classmethod
    def from_generators(
        cls,
        path: str | Path,
        *,
        NUM_CLASSES: int | None = None,
        STATE_ALIGNMENT: int = 16,
    ) -> "BeamConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        actions = payload.get("actions") or payload.get("moves")
        if not actions or not all(isinstance(action, list) for action in actions):
            raise ValueError("generator file must contain a non-empty actions array")
        STATE_LEN = len(actions[0])
        if any(len(action) != STATE_LEN for action in actions):
            raise ValueError("all generator permutations must have the same length")
        if STATE_ALIGNMENT <= 0 or STATE_ALIGNMENT & (STATE_ALIGNMENT - 1):
            raise ValueError("STATE_ALIGNMENT must be a positive power of two")
        STATE_STORAGE_LEN = (
            (STATE_LEN + 4 + STATE_ALIGNMENT - 1) // STATE_ALIGNMENT
        ) * STATE_ALIGNMENT
        return cls(
            MOVE_COUNT=len(actions),
            STATE_LEN=STATE_LEN,
            STATE_STORAGE_LEN=STATE_STORAGE_LEN,
            NUM_CLASSES=NUM_CLASSES or STATE_LEN,
            STATE_ALIGNMENT=STATE_ALIGNMENT,
        )


def load_generators(path: str | Path):
    import jax.numpy as jnp

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = payload.get("actions") or payload.get("moves")
    if not actions:
        raise ValueError("generator file must contain actions or moves")
    return jnp.asarray(actions, dtype=jnp.int32)

