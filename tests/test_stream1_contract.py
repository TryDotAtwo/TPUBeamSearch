import json

import jax.numpy as jnp
import numpy as np

from tpu_beam_search.config import BeamConfig
from tpu_beam_search.stream1_reference import (
    apply_all_moves,
    folded_input_linear,
    quantize_score,
)


def test_config_derives_move_and_state_sizes_from_generators(tmp_path):
    path = tmp_path / "generators.json"
    path.write_text(
        json.dumps(
            {
                "actions": [
                    [1, 0, 2, 3],
                    [0, 2, 1, 3],
                    [3, 1, 2, 0],
                ]
            }
        ),
        encoding="utf-8",
    )

    config = BeamConfig.from_generators(path)

    assert config.MOVE_COUNT == 3
    assert config.STATE_LEN == 4
    assert config.STATE_STORAGE_LEN == 16
    assert config.SCORE_SCALE == 1024
    assert config.SCORE_MAX_KEY == 307200


def test_apply_all_moves_uses_candidate_order_parent_then_move():
    parents = jnp.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=jnp.uint8)
    generators = jnp.array([[1, 0, 2, 3], [3, 2, 1, 0]], dtype=jnp.int32)

    children = apply_all_moves(parents, generators)

    np.testing.assert_array_equal(
        children,
        np.array(
            [
                [20, 10, 30, 40],
                [40, 30, 20, 10],
                [2, 1, 3, 4],
                [4, 3, 2, 1],
            ],
            dtype=np.uint8,
        ),
    )


def test_folded_input_linear_matches_literal_one_hot_result():
    states = jnp.array([[0, 2], [1, 0]], dtype=jnp.uint8)
    # Flattened input rows are (position * NUM_CLASSES + value).
    weight = jnp.array(
        [
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50],
            [6, 60],
        ],
        dtype=jnp.bfloat16,
    )
    bias = jnp.array([7, 70], dtype=jnp.bfloat16)

    actual = folded_input_linear(states, weight, bias, NUM_CLASSES=3)

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[14, 140], [13, 130]], dtype=np.float32),
    )


def test_quantize_score_matches_cuda_clamp_and_round_to_nearest_even():
    scores = jnp.array([-1.0, 0.0, 1.5 / 1024.0, 2.5 / 1024.0, 301.0])

    actual = quantize_score(scores, SCORE_SCALE=1024, SCORE_MAX_Q=300.0)

    np.testing.assert_array_equal(actual, np.array([0, 0, 2, 2, 307200], dtype=np.uint32))
