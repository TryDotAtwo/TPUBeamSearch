import json

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import stream1_layernorm_arithmetic as benchmark
from tpu_beam_search.artgor_reference import artgor_reference_apply
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
    stream1_layernorm_reference_inference,
)


def fixture():
    rng = np.random.default_rng(3)
    def array(shape):
        return jnp.asarray(rng.normal(size=shape) * .2, dtype=jnp.float32)
    params = {
        "encoding": "embedding", "state_size": 2, "num_classes": 3,
        "embed": array((3, 2)), "head_w": array((8, 3)), "head_b": array((3,)),
        "input_stack": [{"lin_w": array((4, 8)), "lin_b": array((8,)),
                         "ln_gamma": jnp.ones(8), "ln_beta": array((8,))}],
        "res_blocks": [],
    }
    for _ in range(2):
        params["res_blocks"].append({
            "lin1_w": array((8, 8)), "lin1_b": array((8,)),
            "lin2_w": array((8, 8)), "lin2_b": array((8,)),
            "ln1_gamma": jnp.ones(8), "ln1_beta": array((8,)),
            "ln2_gamma": jnp.ones(8), "ln2_beta": array((8,)),
        })
    states = jnp.array([[0, 1], [2, 0]], dtype=jnp.uint8)
    architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=2)
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    return params, states, architecture, weights


def test_runtime_original_payload_preserves_output_and_accepts_changed_weights():
    params, states, _, _ = fixture()
    payload, metadata = benchmark.runtime_params(params)
    call = jax.jit(lambda x, p: artgor_reference_apply({**metadata, **p}, x, dtype=jnp.bfloat16))
    np.testing.assert_array_equal(
        artgor_reference_apply({**metadata, **payload}, states, dtype=jnp.bfloat16),
        artgor_reference_apply(params, states, dtype=jnp.bfloat16),
    )
    changed = {**payload, "head_b": payload["head_b"] + 8}
    assert not np.array_equal(call(states, payload), call(states, changed))


def test_jax_prefix_suffix_partition_preserves_model_operation_order():
    _, states, architecture, weights = fixture()
    epsilon = architecture.LAYER_NORM_EPSILON
    hidden = benchmark.reference_prefix(states, weights, architecture)
    expected = stream1_layernorm_reference_inference(states, weights, architecture)
    for depth in range(3):
        actual = benchmark.reference_suffix(hidden, weights.residuals[depth:], weights.output, epsilon)
        np.testing.assert_array_equal(actual, expected)
        if depth < 2:
            hidden = benchmark.reference_block(hidden, weights.residuals[depth], epsilon)


def test_same_suffix_comparison_does_not_attribute_reference_boundary_drift_to_candidate():
    # Segmented oracle deliberately differs; identical candidate/control hidden
    # must have zero replacement error even when both differ from that oracle.
    suffix = jax.jit(lambda x, scale: x * scale)
    hidden = jnp.array([[1., 2.]])
    result = benchmark.compare_same_suffix(suffix, hidden, hidden, (jnp.array(2.),), jnp.zeros((1, 2)))
    assert result["candidate_vs_same_suffix"]["max_abs"] == 0
    assert result["jax_control_vs_segmented"]["max_abs"] == 4


def test_measure_compiles_runtime_arguments_and_writes_real_hlo(tmp_path):
    output, timing, compiled = benchmark.measure(
        lambda x, w: x @ w, jnp.ones((2, 8)), jnp.eye(8),
        warmups=1, repeats=3, hlo_path=tmp_path / "dense.txt",
    )
    np.testing.assert_array_equal(output, np.ones((2, 8)))
    np.testing.assert_array_equal(compiled(jnp.ones((2, 8)), 2 * jnp.eye(8)), 2 * np.ones((2, 8)))
    assert len(timing["samples_s"]) == 3
    assert timing["compile_s"] >= 0 and timing["first_execution_s"] >= 0
    assert (tmp_path / "dense.txt").stat().st_size > 0


def test_checkpoint_rejects_nan_without_destroying_previous_result(tmp_path):
    path = tmp_path / "result.json"
    benchmark.checkpoint(path, {"status": "running"})
    import pytest
    with pytest.raises(ValueError):
        benchmark.checkpoint(path, {"bad": float("nan")})
    assert json.loads(path.read_text())["status"] == "running"


def test_small_suite_records_controls_and_does_not_drop_failed_candidates(tmp_path):
    params, states, architecture, weights = fixture()
    baseline = dict(id="jax-cross", fusion="cross", dense="jax", norm="jax",
                    bm=2, bk=8, bn=8, dense_rounding="late", mean_mode="sum_div",
                    fp32_statistics=False)
    broken = {**baseline, "id": "bad-tile", "dense": "pallas", "bm": 0}
    report = benchmark.run_suite(
        params, artgor_reference_apply, architecture, weights,
        {"legal_test_fixture": np.asarray(states)}, np.ones(states.shape[0], dtype=np.int32) * -1,
        np.array([0, 1, 2]), tmp_path, screen_configs=[baseline, broken],
        full_configs=[baseline], warmups=0, repeats=1, interpret=True,
        screen_batch=2, full_batch=2, promotion_batch=2,
    )
    assert report["status"] == "complete"
    assert any(row["status"] == "error" and row["id"] == "bad-tile" for row in report["screen"])
    assert report["controls"][0]["same_suffix"]["candidate_vs_same_suffix"]["max_abs"] == 0
    assert report["full"][0]["status"] == "ok"
    saved = json.loads((tmp_path / "stream1_layernorm_arithmetic.json").read_text())
    assert saved["status"] == "complete"


def test_unavailable_captured_control_does_not_cancel_runtime_experiments(tmp_path):
    params, states, architecture, weights = fixture()
    def runtime_only(p, x, dtype):
        if not isinstance(p["embed"], jax.core.Tracer):
            raise RuntimeError("captured constants rejected")
        return artgor_reference_apply(p, x, dtype=dtype)
    baseline = dict(id="jax-cross", fusion="cross", dense="jax", norm="jax",
                    bm=2, bk=8, bn=8, dense_rounding="late", mean_mode="sum_div",
                    fp32_statistics=False)
    report = benchmark.run_suite(
        params, runtime_only, architecture, weights,
        {"legal_test_fixture": np.asarray(states)}, np.array([-1, -1]), np.arange(3), tmp_path,
        screen_configs=[baseline], full_configs=[baseline], warmups=0, repeats=1,
        interpret=True, screen_batch=2, full_batch=2, promotion_batch=2,
    )
    assert any(r["id"] == "captured_source" and r["status"] == "error" for r in report["baseline_controls"])
    assert report["full"][0]["status"] == "ok"
    assert report["full_baselines"][0]["typed_quality_vs_original"]["unmasked"]["finite"]
