"""Run the real harness at tiny sizes; never certify target compilation here."""
import importlib
import json

import numpy as np
import pytest

from test_layernorm_followup import model_fixture
from tpu_beam_search.artgor_reference import artgor_reference_apply


def module():
    return importlib.import_module("benchmarks.stream1_execution_boundary")


def test_tiny_bundle_keeps_failed_cases_controls_and_partial_json(tmp_path):
    params, states, arch, weights = model_fixture()
    control = dict(id="jax-none", dense="jax", boundary="none", control=True, bm=8, bk=8, bn=8)
    broken = dict(control, id="bad", boundary="typo", control=False)
    report = module().run_suite(params, artgor_reference_apply, arch, weights,
        {"legal": np.asarray(states), "stress": np.asarray(states[::-1])},
        {"legal": np.array([-1, -1]), "stress": np.array([-1, -1])}, np.arange(3), tmp_path,
        configs=[control, broken], dense_cases=[control], embedding_cases=["jax_flat"],
        screen_batch=2, full_batch=2, promotion_batch=2, warmups=1, repeats=1,
        queue_depth=2, queue_repeats=1, interpret=True)
    assert report["status"] == "complete"
    assert len(report["full"]) == 4
    assert all(r["exact_oracle_on_sample"] for r in report["full"] if r["id"] == "jax-none")
    assert all(r["status"] == "error" and "timing" not in r for r in report["full"] if r["id"] == "bad")
    assert report["promotion_decision"]["selected_for_larger_batch"] == []
    assert all(g["comparison_valid"] for g in report["timing_groups"])
    assert report["observations"] and report["embedding"]
    assert all("mismatch_witnesses" in r for r in report["embedding"] if r["status"] == "ok")
    saved = json.loads((tmp_path / "stream1_execution_boundary.json").read_text())
    assert saved["status"] == "complete"
    assert list((tmp_path / "hlo").glob("*.compiled.txt"))


def test_duplicate_ids_are_rejected_before_overwriting_results(tmp_path):
    params, states, arch, weights = model_fixture()
    config = dict(id="same", dense="jax", boundary="none", control=True)
    with pytest.raises(ValueError, match="unique"):
        module().run_suite(params, artgor_reference_apply, arch, weights,
            {"tiny": np.asarray(states)}, {"tiny": np.array([-1, -1])}, np.arange(3), tmp_path,
            configs=[config, config], screen_batch=2, full_batch=2, promotion_batch=2)


def test_exact_noncontrol_runs_a_genuinely_larger_confirmation(tmp_path):
    params, states, arch, weights = model_fixture()
    config = dict(id="equivalent-test-only", dense="jax", boundary="none", control=False)
    inputs = np.tile(np.asarray(states), (2, 1))
    report = module().run_suite(params, artgor_reference_apply, arch, weights,
        {"legal": inputs, "stress": inputs[::-1]},
        {"legal": np.full(4, -1), "stress": np.full(4, -1)}, np.arange(3), tmp_path,
        configs=[config], dense_cases=[config], embedding_cases=[],
        screen_batch=2, full_batch=2, promotion_batch=4, warmups=1, repeats=1,
        queue_depth=1, queue_repeats=1, interpret=True)
    assert report["promotion_decision"]["selected_for_larger_batch"] == [config["id"]]
    assert report["promotion_decision"]["exact_at_larger_batch"] == [config["id"]]
    assert {r["batch"] for r in report["promotion"]} == {4}
    assert {r["corpus"] for r in report["promotion"]} == {"legal", "stress"}
    assert all(r["eligible_speedup"] > 0 for r in report["promotion"])
    with pytest.raises(FileExistsError, match="new output directory"):
        module().run_suite(params, artgor_reference_apply, arch, weights,
            {"legal": inputs}, {"legal": np.full(4, -1)}, np.arange(3), tmp_path,
            configs=[config], dense_cases=[config], screen_batch=2, full_batch=2, promotion_batch=4)


def test_all_full_cases_failing_is_terminal_error_with_partial_json(tmp_path):
    params, states, arch, weights = model_fixture()
    dense = dict(id="reference", dense="jax", boundary="none", control=True)
    broken = dict(dense, id="bad", boundary="typo")
    with pytest.raises(RuntimeError, match="no full case"):
        module().run_suite(params, artgor_reference_apply, arch, weights,
            {"tiny": np.asarray(states)}, {"tiny": np.full(2, -1)}, np.arange(3), tmp_path,
            configs=[broken], dense_cases=[dense], embedding_cases=[],
            screen_batch=2, full_batch=2, promotion_batch=2, warmups=1, repeats=1,
            queue_depth=1, queue_repeats=1, interpret=True)
    saved = json.loads((tmp_path / "stream1_execution_boundary.json").read_text())
    assert saved["status"] == "error"
    assert saved["full"][0]["status"] == "error"
    assert saved["full_baselines"][0]["status"] == "ok"
