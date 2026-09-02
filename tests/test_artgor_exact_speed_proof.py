import copy
import importlib

import pytest


def _proof_module():
    try:
        return importlib.import_module("benchmarks.artgor_exact_speed_proof")
    except ModuleNotFoundError:
        pytest.fail("artgor_exact_speed_proof benchmark is not implemented")


def test_constant_paired_speedup_has_exact_statistics_and_passes():
    proof = _proof_module()
    stats = proof.paired_speed_statistics(
        baseline_s=[0.030, 0.030, 0.030, 0.030],
        candidate_s=[0.020, 0.020, 0.020, 0.020],
        orders=["AB", "BA", "AB", "BA"],
        threshold=1.5,
        bootstrap_seed=7,
        bootstrap_samples=100,
    )
    assert stats["ratio_of_medians"] == pytest.approx(1.5)
    assert stats["paired_median_speedup"] == pytest.approx(1.5)
    assert stats["minimum_paired_speedup"] == pytest.approx(1.5)
    assert stats["conservative_envelope_speedup"] == pytest.approx(1.5)
    assert stats["paired_bootstrap_lower_99"] == pytest.approx(1.5)
    assert stats["passes_frozen_speed_gate"] is True


def test_one_subthreshold_pair_rejects_even_when_median_is_fast():
    proof = _proof_module()
    stats = proof.paired_speed_statistics(
        baseline_s=[0.040, 0.040, 0.040],
        candidate_s=[0.020, 0.020, 0.030],
        orders=["AB", "BA", "AB"],
        threshold=1.5,
        bootstrap_seed=11,
        bootstrap_samples=100,
    )
    assert stats["ratio_of_medians"] == pytest.approx(2.0)
    assert stats["minimum_paired_speedup"] == pytest.approx(4 / 3)
    assert stats["passes_frozen_speed_gate"] is False


def _passing_report(proof):
    case = {
        "exact": True,
        "timing": {"passes_frozen_speed_gate": True},
    }
    return {
        "context": {
            "runtime": {
                "active_device_count": 8,
                "all_devices_are_tpu": True,
            }
        },
        "cases": {name: copy.deepcopy(case) for name in proof.CASE_NAMES},
    }


def test_final_decision_requires_eight_tpus_exactness_and_every_case_gate():
    proof = _proof_module()
    passing = _passing_report(proof)
    assert proof.decide_speed_proof(passing)["publishable"] is True

    wrong_device_count = copy.deepcopy(passing)
    wrong_device_count["context"]["runtime"]["active_device_count"] = 7
    assert proof.decide_speed_proof(wrong_device_count)["publishable"] is False

    inexact = copy.deepcopy(passing)
    inexact["cases"][proof.CASE_NAMES[0]]["exact"] = False
    assert proof.decide_speed_proof(inexact)["publishable"] is False

    slow = copy.deepcopy(passing)
    slow["cases"][proof.CASE_NAMES[-1]]["timing"][
        "passes_frozen_speed_gate"
    ] = False
    assert proof.decide_speed_proof(slow)["publishable"] is False


def test_final_decision_rejects_missing_case_instead_of_inferring_it():
    proof = _proof_module()
    report = _passing_report(proof)
    del report["cases"][proof.CASE_NAMES[2]]
    decision = proof.decide_speed_proof(report)
    assert decision["publishable"] is False
    assert "all_six_cases_present" in decision["failed_gates"]

