"""CPU contracts for the inference-only 1/8-device promotion harness."""
import importlib
import json

import numpy as np

import pytest

from test_layernorm_followup import model_fixture
from tpu_beam_search.artgor_reference import artgor_reference_apply


def module():
    return importlib.import_module("benchmarks.stream1_inference_8device")


def result(identifier, corpus, latency_ms, *, exact=True, comparable=True,
           device_count=8, local_batch=16384, control=False):
    return dict(
        id=identifier,
        corpus=corpus,
        device_count=device_count,
        local_batch=local_batch,
        global_batch=device_count * local_batch,
        status="ok",
        control=control,
        timing_comparable=comparable,
        exact_oracle_on_sample=exact,
        timing=dict(median_ms=latency_ms),
    )


def test_candidate_matrix_crosses_bm_and_bank_dtype_without_changing_network():
    configs = module().candidate_configs()
    prepacked = [c for c in configs if c["implementation"] == "pallas_prepacked"]
    assert {(c["bm"], c["bank_dtype"]) for c in prepacked} == {
        (64, "bfloat16"), (64, "float32"),
        (128, "bfloat16"), (128, "float32"),
        (256, "bfloat16"), (256, "float32"),
        (512, "bfloat16"), (512, "float32"),
        (1024, "bfloat16"), (1024, "float32"),
        (2048, "bfloat16"), (2048, "float32"),
    }
    assert {c["id"] for c in configs if c["control"]} == {
        "original_runtime_jax", "typed_runtime_jax"}
    assert all(c["network"] == "unchanged_jax_resmlp" for c in configs)


def test_eight_device_winner_must_be_exact_and_faster_on_every_corpus():
    corpora = ("legal", "stress")
    rows = []
    for corpus, baseline_ms in zip(corpora, (12.0, 13.0)):
        rows += [
            result("original_runtime_jax", corpus, baseline_ms, control=True),
            result("wins_both", corpus, baseline_ms - 2.0),
            result("wins_average_only", corpus, 8.0 if corpus == "legal" else 14.0),
        ]
    decision = module().select_eight_device_winner(
        rows, corpus_names=corpora, local_batch=16384)
    assert decision["winner_id"] == "wins_both"
    assert decision["target_achieved"] is True
    assert decision["per_corpus_speedup"] == {
        "legal": 1.2, "stress": pytest.approx(13 / 11)
    }


@pytest.mark.parametrize("change", ["inexact", "unpaired", "missing", "wrong_device", "wrong_batch"])
def test_eight_device_gate_rejects_incomplete_or_incomparable_candidate(change):
    corpora = ("legal", "stress")
    rows = [result("original_runtime_jax", c, 12.0, control=True) for c in corpora]
    candidates = [result("candidate", c, 9.0) for c in corpora]
    if change == "inexact":
        candidates[1]["exact_oracle_on_sample"] = False
    elif change == "unpaired":
        candidates[1]["timing_comparable"] = False
    elif change == "missing":
        candidates.pop()
    elif change == "wrong_device":
        candidates[1]["device_count"] = 1
    else:
        candidates[1]["local_batch"] = 32768
    decision = module().select_eight_device_winner(
        rows + candidates, corpus_names=corpora, local_batch=16384)
    assert decision["winner_id"] is None
    assert decision["target_achieved"] is False


def test_weak_scaling_uses_fixed_local_batch_and_global_throughput():
    one = dict(device_count=1, local_batch=16384, global_batch=16384, timing=dict(median_ms=8.0))
    eight = dict(device_count=8, local_batch=16384, global_batch=131072, timing=dict(median_ms=9.0))
    got = module().weak_scaling(one, eight)
    assert got["one_device_states_per_second"] == 2_048_000.0
    assert got["eight_device_states_per_second"] == pytest.approx(131_072_000 / 9)
    assert got["throughput_speedup"] == pytest.approx(64 / 9)
    assert got["parallel_efficiency"] == pytest.approx(8 / 9)
    with pytest.raises(ValueError, match="local batch"):
        module().weak_scaling(one, {**eight, "local_batch": 8192, "global_batch": 65536})


def test_tiny_inference_bundle_executes_prepacked_path_and_checkpoints_json(tmp_path):
    params, states, arch, weights = model_fixture()
    original = next(c for c in module().candidate_configs()
                    if c["id"] == "original_runtime_jax")
    prepacked = dict(
        next(c for c in module().candidate_configs()
             if c["id"] == "pallas_prepacked_bm64_bfloat16"),
        id="pallas_prepacked_test", bm=8,
    )
    corpora = {"legal": np.asarray(states), "stress": np.asarray(states[::-1])}
    report = module().run_suite(
        params, artgor_reference_apply, arch, weights, corpora,
        {name: np.full(2, -1, np.int32) for name in corpora}, np.arange(4), tmp_path,
        configs=[original, prepacked], screen_local_batch=2,
        confirmation_local_batch=2, device_counts=(1,), warmups=1,
        repeats=1, interpret=True,
    )
    assert report["status"] == "complete"
    assert len(report["measurements"]) == 4
    assert all(row["exact_oracle_on_sample"] for row in report["measurements"])
    assert all(row["device_count"] == 1 and row["global_batch"] == 2
               for row in report["measurements"])
    assert report["decision"]["target_achieved"] is False
    saved = json.loads((tmp_path / "stream1_inference_8device.json").read_text())
    assert saved["status"] == "complete"
    assert list((tmp_path / "hlo").glob("*.compiled.txt"))
