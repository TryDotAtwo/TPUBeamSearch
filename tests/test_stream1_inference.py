import jax.numpy as jnp
import numpy as np
import pytest

from tpu_beam_search.stream1_inference import (
    DenseWeights,
    ResidualWeights,
    Stream1Architecture,
    Stream1Weights,
    make_jitted_stream1_inference,
    stream1_pallas_inference,
    stream1_reference_inference,
    stream1_weights_from_pytorch_state_dict,
)


def tiny_architecture():
    return Stream1Architecture(
        STATE_LEN=2,
        STATE_STORAGE_LEN=4,
        NUM_CLASSES=3,
        HIDDEN1=4,
        HIDDEN2=2,
        RESIDUAL_COUNT=1,
        MOVE_COUNT=3,
    )


def tiny_weights():
    return Stream1Weights(
        input=DenseWeights(
            weight=jnp.array(
                [
                    [1, 0, -1, 2],
                    [0, 2, 1, -1],
                    [3, 1, 0, 0],
                    [1, -1, 2, 1],
                    [2, 0, 1, 2],
                    [-1, 3, 1, 0],
                ],
                dtype=jnp.bfloat16,
            ),
            bias=jnp.array([0, -1, 1, 0], dtype=jnp.bfloat16),
        ),
        hidden=DenseWeights(
            weight=jnp.array(
                [[1, -1], [2, 0], [-1, 1], [0, 2]], dtype=jnp.bfloat16
            ),
            bias=jnp.array([1, -1], dtype=jnp.bfloat16),
        ),
        residuals=(
            ResidualWeights(
                first=DenseWeights(
                    weight=jnp.array([[1, 0], [0, 1]], dtype=jnp.bfloat16),
                    bias=jnp.array([1, -1], dtype=jnp.bfloat16),
                ),
                second=DenseWeights(
                    weight=jnp.array([[1, 1], [-1, 1]], dtype=jnp.bfloat16),
                    bias=jnp.array([0, 1], dtype=jnp.bfloat16),
                ),
            ),
        ),
        output=DenseWeights(
            weight=jnp.array([[1, 0, -1], [0, 2, 1]], dtype=jnp.bfloat16),
            bias=jnp.array([1, -1, 0], dtype=jnp.bfloat16),
        ),
    )


def test_reference_inference_runs_complete_residual_graph_and_ignores_storage_padding():
    states = jnp.array([[0, 2, 99, 88], [1, 0, 77, 66]], dtype=jnp.uint8)

    actual = stream1_reference_inference(states, tiny_weights(), tiny_architecture())

    assert actual.shape == (2, 3)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[7, 25, 7], [1, 9, 5]], dtype=np.float32),
    )


def test_inference_rejects_weights_that_do_not_match_the_static_architecture():
    weights = tiny_weights()._replace(residuals=())

    with pytest.raises(ValueError, match="RESIDUAL_COUNT"):
        stream1_reference_inference(
            jnp.zeros((1, 4), dtype=jnp.uint8), weights, tiny_architecture()
        )


def test_pallas_inference_matches_reference_for_complete_graph_in_interpret_mode():
    states = jnp.array([[0, 2, 99, 88], [1, 0, 77, 66]], dtype=jnp.uint8)
    expected = stream1_reference_inference(states, tiny_weights(), tiny_architecture())

    actual = stream1_pallas_inference(
        states,
        tiny_weights(),
        tiny_architecture(),
        interpret=True,
        bm=2,
        bk_input=2,
        bn_input=2,
        bk_hidden=2,
        bn_hidden=2,
        bk_residual=2,
        bn_residual=2,
        bk_output=2,
        bn_output=2,
    )

    np.testing.assert_array_equal(actual, expected)


def test_jitted_inference_keeps_weights_dynamic_and_architecture_static():
    states = jnp.array([[0, 2, 99, 88], [1, 0, 77, 66]], dtype=jnp.uint8)
    compiled = make_jitted_stream1_inference(
        tiny_architecture(),
        backend="reference",
    )

    first = compiled(states, tiny_weights())
    changed = tiny_weights()._replace(
        output=tiny_weights().output._replace(
            bias=jnp.array([2, -1, 0], dtype=jnp.bfloat16)
        )
    )
    second = compiled(states, changed)

    np.testing.assert_array_equal(
        np.asarray(second, dtype=np.float32) - np.asarray(first, dtype=np.float32),
        np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
    )


def test_jitted_pallas_inference_executes_the_complete_graph():
    states = jnp.array([[0, 2, 99, 88], [1, 0, 77, 66]], dtype=jnp.uint8)
    compiled = make_jitted_stream1_inference(
        tiny_architecture(),
        backend="pallas",
        interpret=True,
        bm=2,
        bk_input=2,
        bn_input=2,
        bk_hidden=2,
        bn_hidden=2,
        bk_residual=2,
        bn_residual=2,
        bk_output=2,
        bn_output=2,
    )

    actual = compiled(states, tiny_weights())

    np.testing.assert_array_equal(
        actual,
        stream1_reference_inference(states, tiny_weights(), tiny_architecture()),
    )


@pytest.mark.parametrize("residual_fusion", ["per_block", "pairs"])
def test_pallas_inference_residual_fusion_modes_preserve_complete_graph(residual_fusion):
    architecture = Stream1Architecture(
        STATE_LEN=2,
        STATE_STORAGE_LEN=4,
        NUM_CLASSES=3,
        HIDDEN1=4,
        HIDDEN2=2,
        RESIDUAL_COUNT=2,
        MOVE_COUNT=3,
    )
    base = tiny_weights()
    weights = base._replace(residuals=(base.residuals[0], base.residuals[0]))
    states = jnp.array([[0, 2, 99, 88], [1, 0, 77, 66]], dtype=jnp.uint8)

    actual = stream1_pallas_inference(
        states,
        weights,
        architecture,
        residual_fusion=residual_fusion,
        interpret=True,
        bm=2,
        bk_input=2,
        bn_input=2,
        bk_hidden=2,
        bn_hidden=2,
        bk_residual=2,
        bn_residual=2,
        bk_output=2,
        bn_output=2,
    )

    np.testing.assert_array_equal(
        np.asarray(actual, dtype=np.float32),
        np.array([[2, 65, 32], [1, 21, 11]], dtype=np.float32),
    )


def test_architecture_rejects_storage_shorter_than_logical_state():
    with pytest.raises(ValueError, match="STATE_STORAGE_LEN"):
        Stream1Architecture(
            STATE_LEN=4,
            STATE_STORAGE_LEN=3,
            NUM_CLASSES=5,
            HIDDEN1=8,
            HIDDEN2=8,
            RESIDUAL_COUNT=0,
            MOVE_COUNT=2,
        )


def test_checkpoint_import_folds_every_batch_norm_and_transposes_linear_weights():
    state_dict = {
        "input_layer.weight": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "input_layer.bias": np.array([1, -1], dtype=np.float32),
        "bn1.weight": np.array([4, 3], dtype=np.float32),
        "bn1.bias": np.array([5, 7], dtype=np.float32),
        "bn1.running_mean": np.array([1, 3], dtype=np.float32),
        "bn1.running_var": np.array([4, 9], dtype=np.float32),
        "hidden_layer.weight": np.eye(2, dtype=np.float32),
        "hidden_layer.bias": np.zeros(2, dtype=np.float32),
        "bn2.weight": np.ones(2, dtype=np.float32),
        "bn2.bias": np.zeros(2, dtype=np.float32),
        "bn2.running_mean": np.zeros(2, dtype=np.float32),
        "bn2.running_var": np.ones(2, dtype=np.float32),
        "residual_blocks.0.fc1.weight": np.eye(2, dtype=np.float32),
        "residual_blocks.0.fc1.bias": np.zeros(2, dtype=np.float32),
        "residual_blocks.0.bn1.weight": np.ones(2, dtype=np.float32),
        "residual_blocks.0.bn1.bias": np.zeros(2, dtype=np.float32),
        "residual_blocks.0.bn1.running_mean": np.zeros(2, dtype=np.float32),
        "residual_blocks.0.bn1.running_var": np.ones(2, dtype=np.float32),
        "residual_blocks.0.fc2.weight": np.eye(2, dtype=np.float32),
        "residual_blocks.0.fc2.bias": np.zeros(2, dtype=np.float32),
        "residual_blocks.0.bn2.weight": np.ones(2, dtype=np.float32),
        "residual_blocks.0.bn2.bias": np.zeros(2, dtype=np.float32),
        "residual_blocks.0.bn2.running_mean": np.zeros(2, dtype=np.float32),
        "residual_blocks.0.bn2.running_var": np.ones(2, dtype=np.float32),
        "output_layer.weight": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "output_layer.bias": np.array([1, 2, 3], dtype=np.float32),
    }
    architecture = Stream1Architecture(
        STATE_LEN=1,
        STATE_STORAGE_LEN=2,
        NUM_CLASSES=2,
        HIDDEN1=2,
        HIDDEN2=2,
        RESIDUAL_COUNT=1,
        MOVE_COUNT=3,
    )

    weights = stream1_weights_from_pytorch_state_dict(
        state_dict, architecture, BN_EPSILON=0.0, dtype=jnp.float32
    )

    np.testing.assert_array_equal(weights.input.weight, [[2, 3], [4, 4]])
    np.testing.assert_array_equal(weights.input.bias, [5, 3])
    np.testing.assert_array_equal(
        weights.output.weight, [[1, 3, 5], [2, 4, 6]]
    )
    assert len(weights.residuals) == 1


def test_architecture_is_derived_from_checkpoint_shapes_not_model_name():
    state_dict = {
        "input_layer.weight": np.zeros((8, 15), dtype=np.float32),
        "hidden_layer.weight": np.zeros((4, 8), dtype=np.float32),
        "residual_blocks.0.fc1.weight": np.zeros((4, 4), dtype=np.float32),
        "residual_blocks.0.fc2.weight": np.zeros((4, 4), dtype=np.float32),
        "residual_blocks.1.fc1.weight": np.zeros((4, 4), dtype=np.float32),
        "residual_blocks.1.fc2.weight": np.zeros((4, 4), dtype=np.float32),
        "output_layer.weight": np.zeros((6, 4), dtype=np.float32),
    }

    architecture = Stream1Architecture.from_pytorch_state_dict(
        state_dict,
        STATE_LEN=3,
        STATE_STORAGE_LEN=8,
        NUM_CLASSES=5,
    )

    assert architecture == Stream1Architecture(
        STATE_LEN=3,
        STATE_STORAGE_LEN=8,
        NUM_CLASSES=5,
        HIDDEN1=8,
        HIDDEN2=4,
        RESIDUAL_COUNT=2,
        MOVE_COUNT=6,
    )
