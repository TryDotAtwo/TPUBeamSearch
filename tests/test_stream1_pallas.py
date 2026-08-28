import jax.numpy as jnp
import numpy as np

from tpu_beam_search.stream1_pallas import (
    pallas_apply_all_moves,
    pallas_dense_linear,
    pallas_embedding_sum_linear,
    pallas_fused_folded_hidden,
    pallas_fused_mlp,
    pallas_fused_residual_block,
    pallas_fused_two_residual_blocks,
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

    assert actual.dtype == jnp.bfloat16
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


def test_pallas_dense_linear_fuses_bias_and_relu_with_bf16_output():
    values = jnp.array([[1, 2, 3, 4], [2, 0, 1, 3]], dtype=jnp.bfloat16)
    weight = jnp.array(
        [[1, -1], [2, 1], [0, 2], [-1, 0]], dtype=jnp.bfloat16
    )
    bias = jnp.array([-1, 2], dtype=jnp.bfloat16)

    actual = pallas_dense_linear(
        values,
        weight,
        bias,
        bm=2,
        bk=2,
        bn=2,
        relu=True,
        interpret=True,
    )

    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[0, 9], [0, 2]], dtype=np.float32),
    )


def test_pallas_fused_folded_hidden_keeps_relu_hidden_between_matmuls():
    states = jnp.array([[0, 2], [1, 0]], dtype=jnp.uint8)
    input_weight = jnp.array(
        [
            [1, 0, -1, 2],
            [0, 2, 1, -1],
            [3, 1, 0, 0],
            [1, -1, 2, 1],
            [2, 0, 1, 2],
            [-1, 3, 1, 0],
        ],
        dtype=jnp.bfloat16,
    )
    input_bias = jnp.array([0, -1, 1, 0], dtype=jnp.bfloat16)
    hidden_weight = jnp.array(
        [[1, -1], [2, 0], [-1, 1], [0, 2]], dtype=jnp.bfloat16
    )
    hidden_bias = jnp.array([1, -1], dtype=jnp.bfloat16)

    actual = pallas_fused_folded_hidden(
        states,
        input_weight,
        input_bias,
        hidden_weight,
        hidden_bias,
        STATE_LEN=2,
        NUM_CLASSES=3,
        bm=2,
        bk_input=2,
        bn_input=2,
        bk_hidden=2,
        bn_hidden=2,
        interpret=True,
    )

    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[4, 4], [0, 2]], dtype=np.float32),
    )


def test_pallas_fused_folded_hidden_accepts_pipeline_controls_in_interpret_mode():
    states = jnp.array([[0, 1]], dtype=jnp.uint8)
    input_weight = jnp.eye(4, dtype=jnp.bfloat16)
    input_bias = jnp.zeros(4, dtype=jnp.bfloat16)
    hidden_weight = jnp.eye(4, dtype=jnp.bfloat16)
    hidden_bias = jnp.zeros(4, dtype=jnp.bfloat16)

    actual = pallas_fused_folded_hidden(
        states,
        input_weight,
        input_bias,
        hidden_weight,
        hidden_bias,
        STATE_LEN=2,
        NUM_CLASSES=2,
        bm=2,
        bk_input=2,
        bn_input=2,
        bk_hidden=2,
        bn_hidden=2,
        pipeline_buffer_count=1,
        pipeline_lookahead=False,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[1, 0, 0, 1]], dtype=np.float32),
    )


def test_pallas_fused_mlp_returns_only_logical_move_logits():
    states = jnp.array([[0, 2], [1, 0]], dtype=jnp.uint8)
    input_weight = jnp.array(
        [
            [1, 0, -1, 2],
            [0, 2, 1, -1],
            [3, 1, 0, 0],
            [1, -1, 2, 1],
            [2, 0, 1, 2],
            [-1, 3, 1, 0],
        ],
        dtype=jnp.bfloat16,
    )
    input_bias = jnp.array([0, -1, 1, 0], dtype=jnp.bfloat16)
    hidden_weight = jnp.array(
        [[1, -1], [2, 0], [-1, 1], [0, 2]], dtype=jnp.bfloat16
    )
    hidden_bias = jnp.array([1, -1], dtype=jnp.bfloat16)
    output_weight = jnp.array(
        [[1, 0, -1], [0, 2, 1]], dtype=jnp.bfloat16
    )
    output_bias = jnp.array([1, -1, 0], dtype=jnp.bfloat16)

    actual = pallas_fused_mlp(
        states,
        input_weight,
        input_bias,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        STATE_LEN=2,
        NUM_CLASSES=3,
        MOVE_COUNT=3,
        bm=2,
        bk_input=2,
        bn_input=2,
        bk_hidden=2,
        bn_hidden=2,
        bk_output=2,
        bn_output=2,
        interpret=True,
    )

    assert actual.shape == (2, 3)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[5, 7, 0], [1, 3, 2]], dtype=np.float32),
    )


def test_pallas_fused_residual_block_keeps_skip_until_second_matmul_finishes():
    values = jnp.array([[4, 4], [0, 2]], dtype=jnp.bfloat16)
    first_weight = jnp.eye(2, dtype=jnp.bfloat16)
    first_bias = jnp.array([1, -1], dtype=jnp.bfloat16)
    second_weight = jnp.array([[1, 1], [-1, 1]], dtype=jnp.bfloat16)
    second_bias = jnp.array([0, 1], dtype=jnp.bfloat16)

    actual = pallas_fused_residual_block(
        values,
        first_weight,
        first_bias,
        second_weight,
        second_bias,
        bm=2,
        bk=2,
        bn=2,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[6, 13], [0, 5]], dtype=np.float32),
    )


def test_pallas_fused_two_residual_blocks_keeps_inter_block_value_in_vmem():
    values = jnp.array([[4, 4], [0, 2]], dtype=jnp.bfloat16)
    first_weight = jnp.eye(2, dtype=jnp.bfloat16)
    first_bias = jnp.array([1, -1], dtype=jnp.bfloat16)
    second_weight = jnp.array([[1, 1], [-1, 1]], dtype=jnp.bfloat16)
    second_bias = jnp.array([0, 1], dtype=jnp.bfloat16)

    actual = pallas_fused_two_residual_blocks(
        values,
        first_weight,
        first_bias,
        second_weight,
        second_bias,
        first_weight,
        first_bias,
        second_weight,
        second_bias,
        bm=2,
        bk=2,
        bn=2,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[1, 33], [0, 11]], dtype=np.float32),
    )
