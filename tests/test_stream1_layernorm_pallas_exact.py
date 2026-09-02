"""Behavioral contracts for the correctness-first all-Pallas ResMLP path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import tpu_beam_search.stream1_layernorm_pallas_exact as pallas_exact_module
from test_layernorm_followup import model_fixture
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    PallasExactConfig,
    make_sharded_pallas_exact_inference,
    pallas_exact_layer_norm_activation,
    pallas_exact_stage_names,
    prepare_pallas_exact_weights,
    stream1_layernorm_pallas_exact_stages,
)


def _tiny_config(**changes):
    values = dict(
        embedding_bm=2,
        input_bm=2,
        input_bk=8,
        input_bn=8,
        residual_bm=2,
        residual_bk=8,
        residual_bn=8,
        head_bm=2,
        head_bk=8,
        head_bn=8,
        dense_rounding="late",
        layernorm_arithmetic="legacy_bf16",
    )
    values.update(changes)
    return PallasExactConfig(**values)


def test_config_rejects_nonpositive_tiles():
    with pytest.raises(ValueError, match="positive"):
        _tiny_config(residual_bk=0)


def test_production_defaults_use_tpu_legal_bias_vector_tiles():
    config = PallasExactConfig()

    assert config.input_bn >= 256
    assert config.residual_bn >= 256


def test_layernorm_block_indices_remain_int32_when_x64_is_enabled():
    with jax.enable_x64():
        matrix = jax.make_jaxpr(pallas_exact_module._matrix_row_index)(
            jnp.int32(0)
        )
        vector = jax.make_jaxpr(pallas_exact_module._vector_zero_index)(
            jnp.int32(0)
        )

    assert [value.aval.dtype for value in matrix.jaxpr.outvars] == [
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int32),
    ]
    assert [value.aval.dtype for value in vector.jaxpr.outvars] == [
        jnp.dtype(jnp.int32),
    ]


def test_prepared_weights_quantize_embedding_once_into_runtime_banks():
    _, _, architecture, weights = model_fixture()

    prepared = prepare_pallas_exact_weights(weights, architecture)

    assert prepared.embedding.low.dtype == jnp.float32
    assert prepared.embedding.high.dtype == jnp.float32
    assert prepared.embedding.low.shape == (1, 128, 128)
    assert prepared.input == weights.input
    assert prepared.residuals == weights.residuals
    assert prepared.output == weights.output


def test_layer_norm_activation_executes_normalization_skip_and_relu_in_pallas():
    values = jnp.asarray(
        [[-2.0, 0.5, 1.5, 3.0], [4.0, -1.0, 2.0, 0.25]],
        dtype=jnp.bfloat16,
    )
    scale = jnp.asarray([0.5, 1.0, 1.5, 2.0], dtype=jnp.bfloat16)
    bias = jnp.asarray([-0.25, 0.5, -0.75, 1.0], dtype=jnp.bfloat16)
    skip = jnp.asarray(
        [[0.0, 0.25, -0.5, 1.0], [1.0, -2.0, 0.5, -0.25]],
        dtype=jnp.bfloat16,
    )
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    expected = jnp.maximum(
        (values - mean) * jax.lax.rsqrt(variance + 1e-5) * scale + bias + skip,
        0,
    ).astype(jnp.bfloat16)

    actual = pallas_exact_layer_norm_activation(
        values,
        scale,
        bias,
        skip=skip,
        add_skip=True,
        relu=True,
        epsilon=1e-5,
        bm=2,
        width_alignment=4,
        arithmetic="legacy_bf16",
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_stage_trace_has_one_pallas_boundary_per_model_operator():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_pallas_exact_weights(weights, architecture)

    trace = stream1_layernorm_pallas_exact_stages(
        states, prepared, architecture, config=_tiny_config(), interpret=True,
    )

    expected_names = pallas_exact_stage_names(architecture)
    assert tuple(stage.name for stage in trace) == expected_names
    assert len(trace) == 4 * architecture.RESIDUAL_COUNT + 4
    assert trace[-1].value.shape == (len(states), architecture.MOVE_COUNT)
    assert bool(jnp.all(jnp.isfinite(trace[-1].value)))


def test_sharded_runner_returns_the_last_diagnostic_stage():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_pallas_exact_weights(weights, architecture)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    runner = make_sharded_pallas_exact_inference(
        architecture,
        mesh=mesh,
        weights_example=prepared,
        config=_tiny_config(),
        interpret=True,
    )
    sharded_states = jax.device_put(
        states, NamedSharding(mesh, P("core", None)),
    )
    replicated_weights = jax.tree.map(
        lambda value: jax.device_put(value, NamedSharding(mesh, P())), prepared,
    )

    actual = runner(sharded_states, replicated_weights)
    expected = stream1_layernorm_pallas_exact_stages(
        states, prepared, architecture, config=_tiny_config(), interpret=True,
    )[-1].value

    assert actual.sharding.spec == P("core", None)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
