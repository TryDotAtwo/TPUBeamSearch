import jax.numpy as jnp
import numpy as np

from tpu_beam_search.stream1_pallas import (
    pallas_apply_all_moves,
    pallas_embedding_sum_linear,
    pallas_folded_input_linear,
)


def test_pallas_apply_moves_preserves_parent_move_candidate_order_in_interpret_mode():
    parents = jnp.array(
        [[10, 20, 30, 40, 0, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0]],
        dtype=jnp.uint8,
    )
    generators = jnp.array(
        [[1, 0, 2, 3, 4, 5, 6, 7], [3, 2, 1, 0, 4, 5, 6, 7]],
        dtype=jnp.int32,
    )

    actual = pallas_apply_all_moves(
        parents,
        generators,
        MOVE_COUNT=2,
        STATE_STORAGE_LEN=8,
        interpret=True,
    )

    np.testing.assert_array_equal(
        actual,
        np.array(
            [
                [20, 10, 30, 40, 0, 0, 0, 0],
                [40, 30, 20, 10, 0, 0, 0, 0],
                [2, 1, 3, 4, 0, 0, 0, 0],
                [4, 3, 2, 1, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_pallas_folded_input_builds_one_hot_tiles_inside_kernel():
    states = jnp.array([[0, 2], [1, 0]], dtype=jnp.uint8)
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

    actual = pallas_folded_input_linear(
        states,
        weight,
        bias,
        STATE_LEN=2,
        NUM_CLASSES=3,
        bm=2,
        bk=2,
        bn=2,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[14, 140], [13, 130]], dtype=np.float32),
    )


def test_pallas_embedding_sum_reads_only_selected_weight_rows():
    states = jnp.array([[0, 2], [1, 0]], dtype=jnp.uint32)
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

    actual = pallas_embedding_sum_linear(
        states,
        weight,
        bias,
        STATE_LEN=2,
        NUM_CLASSES=3,
        bn=2,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[14, 140], [13, 130]], dtype=np.float32),
    )
