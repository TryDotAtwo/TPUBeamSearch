"""Contracts for the exact two-dispatch LayerNorm inference path."""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmarks.final_residual_ops import candidate_final_partition
from test_layernorm_followup import model_fixture
from tpu_beam_search.stream1_layernorm_exact import (
    make_sharded_exact_layernorm_inference,
    prepare_exact_layernorm_inference_weights,
    stream1_layernorm_exact_head,
    stream1_layernorm_exact_prefix,
)
from tpu_beam_search.stream1_layernorm_reference import (
    stream1_layernorm_reference_inference,
)


def test_exact_prefix_and_head_recompose_reference_inference_on_cpu():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)

    hidden = stream1_layernorm_exact_prefix(
        states, prepared, architecture, bm=2, interpret=True,
    )
    actual = stream1_layernorm_exact_head(hidden, prepared, architecture)

    np.testing.assert_array_equal(
        actual,
        stream1_layernorm_reference_inference(states, weights, architecture),
    )
    assert hidden.shape == (len(states), architecture.HIDDEN2)


def test_production_stages_match_the_benchmarked_winner_formulas():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)
    benchmark_weights = weights._replace(embedding=prepared.embedding)
    benchmark_prefix, benchmark_suffix = candidate_final_partition(
        dict(
            embedding="pallas_banked_prepacked",
            dense="jax",
            norm="jax",
            input_boundary="none",
            final_barrier="none",
            bm=2,
        ),
        architecture,
        cut="after_final_block",
        interpret=True,
    )

    expected_hidden = benchmark_prefix(states, benchmark_weights)
    actual_hidden = stream1_layernorm_exact_prefix(
        states, prepared, architecture, bm=2, interpret=True,
    )

    np.testing.assert_array_equal(actual_hidden, expected_hidden)
    np.testing.assert_array_equal(
        stream1_layernorm_exact_head(actual_hidden, prepared, architecture),
        benchmark_suffix(expected_hidden, benchmark_weights),
    )


def test_exact_head_keeps_output_weights_dynamic():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)
    hidden = stream1_layernorm_exact_prefix(
        states, prepared, architecture, bm=2, interpret=True,
    )
    changed = prepared._replace(
        output=prepared.output._replace(bias=prepared.output.bias + 4),
    )

    compiled = jax.jit(
        lambda values, dynamic_weights: stream1_layernorm_exact_head(
            values, dynamic_weights, architecture,
        )
    )
    original = compiled(hidden, prepared)
    actual = compiled(hidden, changed)

    np.testing.assert_array_equal(
        actual,
        stream1_layernorm_exact_head(hidden, changed, architecture),
    )
    assert not np.array_equal(actual, original)


def test_sharded_exact_runner_preserves_a_real_dispatch_boundary():
    _, states, architecture, weights = model_fixture()
    prepared = prepare_exact_layernorm_inference_weights(weights, architecture)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    runner = make_sharded_exact_layernorm_inference(
        architecture,
        mesh=mesh,
        weights_example=prepared,
        bm=2,
        interpret=True,
    )
    sharded_states = jax.device_put(
        states, NamedSharding(mesh, P("core", None)),
    )
    replicated_weights = jax.tree.map(
        lambda value: jax.device_put(value, NamedSharding(mesh, P())),
        prepared,
    )

    hidden = runner.prefix(sharded_states, replicated_weights)
    actual = runner(sharded_states, replicated_weights)

    assert hidden.sharding.spec == P("core", None)
    np.testing.assert_array_equal(
        actual,
        runner.suffix(hidden, replicated_weights),
    )
