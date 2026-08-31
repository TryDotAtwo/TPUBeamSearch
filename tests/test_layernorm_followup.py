"""Follow-up protocol tests; CPU execution is not a TPU compile check."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import stream1_layernorm_followup as bench
from tpu_beam_search.artgor_reference import artgor_reference_apply
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import layernorm_stream1_weights_from_artgor_params


def model_fixture():
    rng = np.random.default_rng(30)
    def a(shape):
        return jnp.asarray(rng.normal(size=shape) * .2, jnp.float32)
    params = dict(encoding="embedding", state_size=2, num_classes=3,
                  embed=a((3, 2)), head_w=a((8, 3)), head_b=a((3,)),
                  input_stack=[dict(lin_w=a((4, 8)), lin_b=a((8,)),
                                    ln_gamma=jnp.ones(8), ln_beta=a((8,)))],
                  res_blocks=[dict(lin1_w=a((8, 8)), lin1_b=a((8,)),
                                   lin2_w=a((8, 8)), lin2_b=a((8,)),
                                   ln1_gamma=jnp.ones(8), ln1_beta=a((8,)),
                                   ln2_gamma=jnp.ones(8), ln2_beta=a((8,)))])
    arch = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=2)
    weights = layernorm_stream1_weights_from_artgor_params(params, arch)
    return params, jnp.array([[0, 1], [2, 0]], jnp.uint8), arch, weights


def test_full_jax_control_preserves_graph_order_and_runtime_head():
    params, states, arch, weights = model_fixture()
    config = dict(id="control", dense="jax", norm="jax", control=True, bm=2, bk=8, bn=8)
    call = jax.jit(bench.full_call(config, arch, interpret=True))
    expected = jax.jit(lambda x, p: artgor_reference_apply(
        {**params, **p}, x, dtype=jnp.bfloat16))(states, {"head_b": params["head_b"]})
    np.testing.assert_array_equal(call(states, weights), expected)
    changed = weights._replace(output=weights.output._replace(bias=weights.output.bias + 8))
    assert not np.array_equal(call(states, weights), call(states, changed))


def test_full_late_dense_keeps_jax_prefix_head_and_all_residuals():
    from benchmarks.stream1_layernorm_arithmetic import full_call as previous_full_call
    _, states, arch, weights = model_fixture()
    config = dict(id="late", dense="late", norm="jax", control=False, bm=2, bk=8, bn=8)
    previous = dict(config, fusion="cross", dense="pallas", dense_rounding="late")
    got = jax.jit(bench.full_call(config, arch, interpret=True))(states, weights)
    expected = jax.jit(previous_full_call(previous, arch, interpret=True))(states, weights)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("reference_offset,exact", [(0, False), (1, True)])
def test_quality_does_not_suppress_diagnostic_profile_or_enable_sample_speedup(tmp_path, reference_offset, exact):
    x = jnp.ones((2, 3))
    compiled = jax.jit(lambda a: a + 1).lower(x).compile()
    calls = []
    def profile(call, *args, directory, iterations):
        calls.append(np.asarray(call(*args)))
        return {"label": "diagnostic_only", "directory": str(directory)}
    result = bench.evaluate_full_case(
        jnp.ones((2, 3)) + reference_offset, compiled(x), np.ones((2, 3), bool),
        compiled=compiled, arguments=(x,), directory=tmp_path,
        profile=True, profile_function=profile)
    assert result["exact_oracle_on_sample"] is exact
    assert result["eligible_speedup"] is None
    assert result["profile"]["label"] == "diagnostic_only"
    assert len(calls) == 1


def test_gate_never_promotes_control_missing_corpus_or_nonexact_output():
    configs = [dict(id="control", control=True), dict(id="good", control=False),
               dict(id="missing", control=False), dict(id="wrong", control=False)]
    rows = [dict(id=c, corpus=name, status="ok", exact_oracle_on_sample=c != "wrong",
                 timing={"median_ms": 2.}, timing_comparable=True)
            for c in ("control", "good", "wrong") for name in ("legal", "stress")]
    rows.append(dict(id="missing", corpus="legal", status="ok", exact_oracle_on_sample=True,
                     timing={"median_ms": .1}, timing_comparable=True))
    assert [c["id"] for c in bench.promotion_candidates(configs, rows, ("legal", "stress"))] == ["good"]


def test_tiny_bundle_preserves_errors_and_exact_control(tmp_path):
    params, states, arch, weights = model_fixture()
    control = dict(id="jax-control", dense="jax", norm="jax", control=True, bm=2, bk=8, bn=8)
    broken = dict(control, id="bad-tile", dense="late", bm=0, control=False)
    report = bench.run_suite(
        params, artgor_reference_apply, arch, weights,
        {"tiny": np.asarray(states)}, {"tiny": np.array([-1, -1])}, np.arange(3), tmp_path,
        configs=[control, broken], screen_batch=2, full_batch=2, promotion_batch=2,
        warmups=1, repeats=1, queue_depth=2, queue_repeats=1,
        interpret=True, synthetic_probes=False)
    assert report["status"] == "complete"
    assert any(r["id"] == "bad-tile" and r["status"] == "error" for r in report["full"])
    control_row = next(r for r in report["full"] if r["id"] == "jax-control")
    assert control_row["exact_oracle_on_sample"]
    assert control_row["queued"]["label"] == "queued_same_executable_not_real_scan"
    assert report["promotion_decision"]["selected_for_larger_batch"] == []
    saved = json.loads((tmp_path / "stream1_layernorm_followup.json").read_text())
    assert saved["status"] == "complete"
    assert list((tmp_path / "hlo").glob("*.stablehlo.txt"))


def test_comparison_group_records_failure_and_retains_unpaired_diagnostics():
    x = jnp.ones(2)
    compiled = jax.jit(lambda v: v + 1).lower(x).compile()
    group = bench.measure_comparison_group(
        {"good": (compiled, (x,)), "wrong-shape": (compiled, (jnp.ones(3),))},
        warmups=1, repeats=1)
    assert group["status"] == "error"
    assert group["comparison_valid"] is False
    assert group["error"]["type"] == "TypeError"
    assert group["label"] == "unpaired_diagnostic_after_group_failure"
    assert group["cases"]["good"]["median_ms"] >= 0
    assert group["case_errors"]["wrong-shape"]["type"] == "TypeError"


def test_group_failure_invalidates_promotion_even_when_q_is_exact():
    configs = [dict(id="candidate", control=False)]
    rows = [dict(id="candidate", corpus=name, status="ok", exact_oracle_on_sample=True,
                 timing={"median_ms": 2.}, timing_comparable=name == "legal")
            for name in ("legal", "stress")]
    assert bench.promotion_candidates(configs, rows, ("legal", "stress")) == []


def test_eligible_speedups_wait_for_all_corpora_and_exclude_controls():
    configs = [dict(id="candidate", control=False), dict(id="control", control=True)]
    rows = [dict(id=identifier, corpus=name, batch=2, status="ok",
                 exact_oracle_on_sample=identifier == "control" or name == "legal",
                 timing={"median_ms": 2.}, timing_comparable=True,
                 eligible_speedup=None, eligible_speedup_vs_typed=None)
            for identifier in ("candidate", "control") for name in ("legal", "stress")]
    baselines = [dict(id=identifier, corpus=name, batch=2, status="ok",
                      timing={"median_ms": 4.}, timing_comparable=True,
                      quality_vs_original={"unmasked": {"finite": True, "exact_fraction": 1.}})
                 for identifier in ("original_runtime", "typed_runtime")
                 for name in ("legal", "stress")]
    bench.finalize_eligible_speedups(configs, rows, baselines, ("legal", "stress"), batch=2)
    assert all(row["eligible_speedup"] is None for row in rows)
    assert all(row["eligible_speedup_vs_typed"] is None for row in rows)

    # The gate consumes recorded real quality results. This isolated reporting
    # test changes the gate input, not the model or the suite's Q acceptance.
    rows[1]["exact_oracle_on_sample"] = True
    bench.finalize_eligible_speedups(configs, rows, baselines, ("legal", "stress"), batch=2)
    assert [row["eligible_speedup"] for row in rows] == [2., 2., None, None]
    rows[1]["timing_comparable"] = False
    bench.finalize_eligible_speedups(configs, rows, baselines, ("legal", "stress"), batch=2)
    assert all(row["eligible_speedup"] is None for row in rows)


def test_tiny_group_failure_is_saved_without_aborting_other_corpora(tmp_path, monkeypatch):
    params, states, arch, weights = model_fixture()
    config = dict(id="jax-control", dense="jax", norm="jax", control=True, bm=2, bk=8, bn=8)
    real_paired = bench.paired_interleaved_measure
    failed = False

    def fail_one_full_group(cases, **options):
        nonlocal failed
        if "original_runtime" in cases and len(cases) > 1 and not failed:
            failed = True
            raise RuntimeError("one comparison group dispatch failed")
        return real_paired(cases, **options)

    # Fault only the dispatch boundary. Compilation, Q decisions, fallback
    # execution, and every unaffected group's timing remain real CPU work.
    monkeypatch.setattr(bench, "paired_interleaved_measure", fail_one_full_group)
    report = bench.run_suite(
        params, artgor_reference_apply, arch, weights,
        {"legal": np.asarray(states), "stress": np.asarray(states)},
        {"legal": np.array([-1, -1]), "stress": np.array([-1, -1])}, np.arange(3), tmp_path,
        configs=[config], screen_batch=2, full_batch=2, promotion_batch=2,
        warmups=1, repeats=1, queue_depth=2, queue_repeats=1,
        interpret=True, synthetic_probes=False)
    assert report["status"] == "complete"
    saved = json.loads((tmp_path / "stream1_layernorm_followup.json").read_text())
    failed_groups = [group for group in saved["timing_groups"] if group["status"] == "error"]
    assert len(failed_groups) == 1
    assert failed_groups[0]["corpus"] == "legal"
    legal, stress = report["full"]
    assert legal["timing_comparable"] is False
    assert stress["timing_comparable"] is True
    assert legal["eligible_speedup"] is None
    assert legal["queued"]["label"] == "queued_same_executable_not_real_scan"
    production_groups = [group for group in report["timing_groups"]
                         if group["scope"] in ("dense", "layernorm")]
    assert len(production_groups) == 4
    assert all(len(group["execution_order"][0]) >= 2 for group in production_groups)
    assert all("queued" in row for row in report["operators"] if row["status"] == "ok")
    assert all("queued" in row for row in report["screen"] if row["status"] == "ok")
    assert list((tmp_path / "hlo").glob("*-same-suffix.compiled.txt"))


def test_synthetic_bm_variants_share_values_and_normalization():
    seen = {}

    def record(section, identifier, action):
        action()

    def capture(call, args, expected, identifier):
        seen[identifier] = args

    bench.run_synthetic_probes(record, capture, interpret=True)
    for width in (1024, 130):
        for prefix in ("predicate", "ln"):
            suffix = "bf16-broadcast" if prefix == "predicate" else "legacy_bf16-all"
            first = seen[f"{prefix}-bm128-w{width}-{suffix}"]
            second = seen[f"{prefix}-bm256-w{width}-{suffix}"]
            assert first[0] is second[0]
            if prefix == "ln":
                assert first[1] is second[1]
