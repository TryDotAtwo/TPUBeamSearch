"""Behavioral contracts for the correctness-first all-Pallas ResMLP path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import tpu_beam_search.stream1_layernorm_pallas_exact as pallas_exact_module
from tpu_beam_search.stream1_layernorm_pallas_attribution import (
    PallasLayerNormArithmetic,
    pallas_layernorm_probe,
)
from test_layernorm_followup import model_fixture
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    PallasExactConfig,
    make_sharded_pallas_exact_inference,
    pallas_exact_input_block,
    pallas_exact_layer_norm_activation,
    pallas_exact_residual_block,
    pallas_exact_custom_call_count,
    pallas_exact_stage_names,
    prepare_pallas_exact_weights,
    stream1_layernorm_pallas_exact_stages,
    pallas_fully_materialized_layernorm_checkpoints,
    _layer_norm_activation_math,
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


@pytest.mark.parametrize("add_skip", [False, True])
def test_split_mean_layernorm_matches_hlo_mixed_interpret(add_skip):
    values = jnp.arange(256, dtype=jnp.float32).reshape(2, 128).astype(jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.25, 0.25, 128).astype(jnp.bfloat16)
    skip = jnp.flip(values, axis=1) if add_skip else None
    expected = pallas_exact_layer_norm_activation(
        values, scale, bias, skip=skip, add_skip=add_skip, relu=True,
        bm=2, arithmetic="hlo_mixed", interpret=True,
    )
    actual = pallas_exact_layer_norm_activation(
        values, scale, bias, skip=skip, add_skip=add_skip, relu=True,
        bm=2, arithmetic="split_mean_hlo_mixed", interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_split_mean_custom_call_contract_adds_one_dispatch_per_layernorm():
    _, _, architecture, _ = model_fixture()
    ordinary = _tiny_config(layernorm_arithmetic="hlo_mixed")
    split = _tiny_config(layernorm_arithmetic="split_mean_hlo_mixed")
    semantic = len(pallas_exact_stage_names(architecture))
    layernorms = 1 + 2 * architecture.RESIDUAL_COUNT
    assert pallas_exact_custom_call_count(architecture, ordinary) == semantic
    assert pallas_exact_custom_call_count(architecture, split) == semantic + layernorms


@pytest.mark.parametrize("add_skip", [False, True])
def test_fully_materialized_layernorm_matches_hlo_mixed_interpret(add_skip):
    values = jnp.arange(256, dtype=jnp.float32).reshape(2, 128).astype(jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.25, 0.25, 128).astype(jnp.bfloat16)
    skip = jnp.flip(values, axis=1) if add_skip else None
    expected = pallas_exact_layer_norm_activation(
        values, scale, bias, skip=skip, add_skip=add_skip, relu=True,
        bm=2, arithmetic="hlo_mixed", interpret=True,
    )
    actual = pallas_exact_layer_norm_activation(
        values, scale, bias, skip=skip, add_skip=add_skip, relu=True,
        bm=2, arithmetic="fully_materialized_hlo_mixed", interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_fully_materialized_contract_uses_five_calls_per_layernorm():
    _, _, architecture, _ = model_fixture()
    config = _tiny_config(layernorm_arithmetic="fully_materialized_hlo_mixed")
    semantic = len(pallas_exact_stage_names(architecture))
    layernorms = 1 + 2 * architecture.RESIDUAL_COUNT
    assert pallas_exact_custom_call_count(architecture, config) == semantic + 4 * layernorms


def test_monolithic_fp32_variance_matches_attribution_winner_in_interpret():
    values = jnp.arange(256, dtype=jnp.float32).reshape(2, 128).astype(jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.25, 0.25, 128).astype(jnp.bfloat16)
    expected = pallas_layernorm_probe(
        values, scale, bias, checkpoint="relu", bm=2, interpret=True,
        arithmetic=PallasLayerNormArithmetic(variance_bf16=False),
    )
    actual = pallas_exact_layer_norm_activation(
        values, scale, bias, relu=True, bm=2,
        arithmetic="monolithic_fp32_variance", interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_monolithic_fp32_variance_keeps_one_call_per_semantic_stage():
    _, _, architecture, _ = model_fixture()
    config = _tiny_config(layernorm_arithmetic="monolithic_fp32_variance")
    assert pallas_exact_custom_call_count(architecture, config) == len(
        pallas_exact_stage_names(architecture)
    )


def test_aligned_monolithic_math_has_no_predicate_or_select_in_jaxpr():
    values = jnp.ones((2, 128), jnp.bfloat16)
    scale = jnp.ones((128,), jnp.bfloat16)
    bias = jnp.zeros((128,), jnp.bfloat16)
    skip = jnp.zeros((2, 128), jnp.bfloat16)
    closed = jax.make_jaxpr(lambda x, s, b, k: _layer_norm_activation_math(
        x, s, b, k, logical_width=128, epsilon=1e-5,
        arithmetic="monolithic_fp32_variance", add_skip=False, relu=True,
        mask_padding=False,
    ))(values, scale, bias, skip)
    primitives = {equation.primitive.name for equation in closed.jaxpr.eqns}
    assert "lt" not in primitives
    assert "select_n" not in primitives


def test_no_skip_monolithic_mode_reuses_the_tpu_proven_exact_probe(monkeypatch):
    values = jnp.ones((2, 128), jnp.bfloat16)
    scale = jnp.ones((128,), jnp.bfloat16)
    bias = jnp.zeros((128,), jnp.bfloat16)
    sentinel = jnp.full_like(values, 7)
    calls = []

    def exact_probe(*args, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(pallas_exact_module, "pallas_layernorm_probe", exact_probe)
    actual = pallas_exact_layer_norm_activation(
        values, scale, bias, relu=True,
        arithmetic="monolithic_fp32_variance", interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(sentinel))
    assert calls[0]["checkpoint"] == "relu"
    assert calls[0]["arithmetic"].variance_bf16 is False


def test_fully_materialized_checkpoints_expose_the_executed_boundaries():
    values = jnp.arange(256, dtype=jnp.float32).reshape(2, 128).astype(jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, 128).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.25, 0.25, 128).astype(jnp.bfloat16)
    checkpoints = pallas_fully_materialized_layernorm_checkpoints(
        values, scale, bias, relu=True, bm=2, interpret=True,
    )
    assert tuple(checkpoints) == (
        "mean", "centered", "variance", "invstd", "affine_relu",
    )
    expected = pallas_exact_layer_norm_activation(
        values, scale, bias, relu=True, bm=2,
        arithmetic="fully_materialized_hlo_mixed", interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(checkpoints["affine_relu"]), np.asarray(expected),
    )


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


def test_isolated_input_and_residual_blocks_match_the_stage_trace():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_pallas_exact_weights(weights, architecture)
    config = _tiny_config(layernorm_arithmetic="monolithic_fp32_variance")

    trace = stream1_layernorm_pallas_exact_stages(
        states, prepared, architecture, config=config, interpret=True,
    )
    input_hidden = pallas_exact_input_block(
        states, prepared, architecture, config=config, interpret=True,
    )
    block_hidden = pallas_exact_residual_block(
        input_hidden, prepared.residuals[0], architecture,
        config=config, interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(input_hidden), np.asarray(trace[2].value))
    np.testing.assert_array_equal(np.asarray(block_hidden), np.asarray(trace[6].value))


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
