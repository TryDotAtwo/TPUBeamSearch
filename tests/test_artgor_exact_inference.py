import jax
import numpy as np
import tpu_beam_search
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmarks import stream1_exact_inference_frontier as frontier
from test_layernorm_followup import model_fixture
from tpu_beam_search.artgor_exact_inference import (
    ArtgorExactConfig,
    choose_artgor_inference_engine,
    prepare_artgor_exact_inference_from_weights,
    prepare_artgor_pallas_exact_inference_from_weights,
)
from tpu_beam_search.stream1_layernorm_exact import (
    stream1_layernorm_exact_prefix,
)
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    PallasExactConfig,
    stream1_layernorm_pallas_exact_inference,
)


def test_selected_config_is_frozen_and_invalid_overrides_fail():
    assert ArtgorExactConfig() == ArtgorExactConfig(
        prefix_bm=4096,
        head_bm=256,
        head_bk=1024,
        head_bn=128,
        dense_rounding="late",
        inference_chunk=32768,
        parent_chunk=131072,
    )
    with np.testing.assert_raises(ValueError):
        ArtgorExactConfig(inference_chunk=24576).validate()


def test_all_pallas_engine_is_exported_from_the_package_api():
    assert tpu_beam_search.PallasExactConfig is PallasExactConfig
    assert (
        tpu_beam_search.prepare_artgor_pallas_exact_inference_from_weights
        is prepare_artgor_pallas_exact_inference_from_weights
    )


def test_unsupported_modes_choose_explicit_jax_fallback():
    assert (
        choose_artgor_inference_engine("exact_split", None, 0.0).selected
        == "exact_split"
    )
    decision = choose_artgor_inference_engine(
        "exact_split", ["q555_6k.pt"], 0.0
    )
    assert decision.selected == "original_jax"
    assert "BLEND_CHECKPOINTS" in decision.reason

    decision = choose_artgor_inference_engine("exact_split", None, 0.25)
    assert decision.selected == "original_jax"
    assert "QV_CONSISTENCY" in decision.reason

    assert (
        choose_artgor_inference_engine("pallas_exact", None, 0.0).selected
        == "pallas_exact"
    )
    decision = choose_artgor_inference_engine(
        "pallas_exact", ["q555_6k.pt"], 0.0
    )
    assert decision.selected == "original_jax"
    assert "BLEND_CHECKPOINTS" in decision.reason

    decision = choose_artgor_inference_engine("pallas_exact", None, 0.25)
    assert decision.selected == "original_jax"
    assert "QV_CONSISTENCY" in decision.reason


def test_interpreted_engine_matches_selected_prefix_and_head_formulas():
    _, states, architecture, weights = model_fixture()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    engine, prepared = prepare_artgor_exact_inference_from_weights(
        weights,
        architecture,
        mesh=mesh,
        config=ArtgorExactConfig(
            prefix_bm=2,
            head_bm=2,
            head_bk=8,
            head_bn=3,
            inference_chunk=2,
            parent_chunk=4,
        ),
        interpret=True,
    )
    states_d = jax.device_put(
        states, NamedSharding(mesh, P("core", None))
    )
    prepared_d = jax.tree.map(
        lambda value: jax.device_put(value, NamedSharding(mesh, P())),
        prepared,
    )

    hidden = engine.prefix(states_d, prepared_d)
    actual = engine.head(hidden, prepared_d)
    benchmark_prefix = frontier._mapped(
        lambda value, runtime_weights: stream1_layernorm_exact_prefix(
            value,
            runtime_weights,
            architecture,
            bm=2,
            interpret=True,
        ),
        mesh=mesh,
        weights_example=prepared,
    )
    benchmark_head = frontier._mapped(
        frontier.head_call(
            {
                "backend": "pallas",
                "bm": 2,
                "bk": 8,
                "bn": 3,
                "dense_rounding": "late",
            },
            architecture,
            interpret=True,
        ),
        mesh=mesh,
        weights_example=prepared,
    )
    expected_hidden = benchmark_prefix(states_d, prepared_d)
    expected = benchmark_head(expected_hidden, prepared_d)

    np.testing.assert_array_equal(hidden, expected_hidden)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(engine(states_d, prepared_d), actual)


def test_interpreted_artgor_pallas_exact_engine_matches_direct_all_pallas_call():
    _, states, architecture, weights = model_fixture()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    config = PallasExactConfig(
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
        layernorm_arithmetic="legacy_bf16",
    )
    engine, prepared = prepare_artgor_pallas_exact_inference_from_weights(
        weights,
        architecture,
        mesh=mesh,
        config=config,
        interpret=True,
    )
    states_d = jax.device_put(states, NamedSharding(mesh, P("core", None)))
    prepared_d = jax.tree.map(
        lambda value: jax.device_put(value, NamedSharding(mesh, P())), prepared,
    )

    actual = engine(states_d, prepared_d)
    expected = stream1_layernorm_pallas_exact_inference(
        states,
        prepared,
        architecture,
        config=config,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
