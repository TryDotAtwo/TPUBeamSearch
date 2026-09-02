import copy
import importlib

import numpy as np
import pytest

from test_layernorm_followup import model_fixture
from tpu_beam_search.stream1_layernorm_reference import (
    stream1_layernorm_reference_inference,
)


def _module():
    try:
        return importlib.import_module("benchmarks.artgor_pallas_exact_diagnostic")
    except ModuleNotFoundError:
        pytest.fail("all-Pallas diagnostic benchmark is not implemented")


def test_first_stage_mismatch_reports_the_actual_operator_boundary():
    diagnostic = _module()
    reference = [
        ("embedding", np.asarray([[1, 2]], dtype=np.uint16)),
        ("input.dense", np.asarray([[3, 4]], dtype=np.uint16)),
        ("input.layernorm_relu", np.asarray([[5, 6]], dtype=np.uint16)),
    ]
    candidate = [
        ("embedding", np.asarray([[1, 2]], dtype=np.uint16)),
        ("input.dense", np.asarray([[3, 9]], dtype=np.uint16)),
        ("input.layernorm_relu", np.asarray([[8, 8]], dtype=np.uint16)),
    ]

    result = diagnostic.compare_stage_sequences(reference, candidate)

    assert result["all_stages_exact"] is False
    assert result["first_mismatch"] == "input.dense"
    assert result["stages"][0]["exact"] is True
    assert result["stages"][1]["mismatch_count"] == 1


def test_stage_comparison_rejects_name_or_shape_drift():
    diagnostic = _module()
    with pytest.raises(ValueError, match="stage names"):
        diagnostic.compare_stage_sequences(
            [("a", np.zeros((1, 2), dtype=np.uint16))],
            [("b", np.zeros((1, 2), dtype=np.uint16))],
        )


def test_reference_stage_sequence_ends_at_the_unchanged_semantic_model():
    diagnostic = _module()
    _, states, architecture, weights = model_fixture()

    stages = diagnostic.reference_stage_sequence(states, weights, architecture)

    assert tuple(name for name, _ in stages) == diagnostic.pallas_exact_stage_names(
        architecture
    )
    np.testing.assert_array_equal(
        np.asarray(stages[-1][1]),
        np.asarray(
            stream1_layernorm_reference_inference(states, weights, architecture)
        ),
    )
    with pytest.raises(ValueError, match="shape"):
        diagnostic.compare_stage_sequences(
            [("a", np.zeros((1, 2), dtype=np.uint16))],
            [("a", np.zeros((2, 1), dtype=np.uint16))],
        )


def test_first_diagnostic_freezes_bk128_and_whole_k_candidates():
    diagnostic = _module()

    configs = diagnostic.candidate_configs()

    assert tuple(configs) == ("pallas_exact_bk128", "pallas_exact_bk1024")
    assert configs["pallas_exact_bk128"].input_bk == 128
    assert configs["pallas_exact_bk128"].residual_bk == 128
    assert configs["pallas_exact_bk1024"].input_bk == 1024
    assert configs["pallas_exact_bk1024"].residual_bk == 1024
    assert all(config.input_bn >= 256 for config in configs.values())
    assert all(config.residual_bn >= 256 for config in configs.values())
    assert all(config.dense_rounding == "late" for config in configs.values())
    assert all(
        config.layernorm_arithmetic == "hlo_mixed"
        for config in configs.values()
    )


def test_stage_count_contract_maps_only_four_n_plus_four_models():
    diagnostic = _module()

    names = diagnostic.pallas_exact_stage_names_from_count(44)

    assert len(names) == 44
    assert names[0] == "embedding"
    assert names[-1] == "head.dense"
    assert names[-2] == "residual.9.layernorm2_skip_relu"
    with pytest.raises(ValueError, match="4\\*N\\+4"):
        diagnostic.pallas_exact_stage_names_from_count(43)


def test_hlo_audit_requires_all_44_custom_calls_and_rejects_jax_model_ops():
    diagnostic = _module()
    stage_names = tuple(
        ["embedding", "input.dense", "input.layernorm_relu"]
        + [
            name
            for block in range(10)
            for name in (
                f"residual.{block}.dense1",
                f"residual.{block}.layernorm1_relu",
                f"residual.{block}.dense2",
                f"residual.{block}.layernorm2_skip_relu",
            )
        ]
        + ["head.dense"]
    )
    clean = "\n".join(
        f'custom_call @tpu_custom_call {{stage = "{name}"}}' for name in stage_names
    )
    passed = diagnostic.audit_all_pallas_hlo(clean, stage_names)
    assert passed["passes"] is True
    assert passed["custom_call_count"] == 44

    contaminated = clean + "\n%0 = stablehlo.dot_general %arg0, %arg1"
    failed = diagnostic.audit_all_pallas_hlo(contaminated, stage_names)
    assert failed["passes"] is False
    assert "stablehlo.dot_general" in failed["forbidden_operations"]


def _passing_report(diagnostic):
    case = {
        "all_stages_exact": True,
        "full_output_exact": True,
        "timing": {"passes_frozen_speed_gate": True},
    }
    return {
        "context": {
            "runtime": {
                "active_device_count": 8,
                "all_devices_are_tpu": True,
            }
        },
        "hlo_audit": {"passes": True},
        "cases": {
            name: copy.deepcopy(case) for name in diagnostic.CASE_NAMES
        },
    }


def test_final_gate_requires_exact_stages_clean_hlo_and_speed_vs_hybrid():
    diagnostic = _module()
    passing = _passing_report(diagnostic)
    assert diagnostic.decide_pallas_exact(passing)["promote"] is True

    inexact = copy.deepcopy(passing)
    inexact["cases"][diagnostic.CASE_NAMES[0]]["all_stages_exact"] = False
    assert diagnostic.decide_pallas_exact(inexact)["promote"] is False

    dirty_hlo = copy.deepcopy(passing)
    dirty_hlo["hlo_audit"]["passes"] = False
    assert diagnostic.decide_pallas_exact(dirty_hlo)["promote"] is False

    slow = copy.deepcopy(passing)
    slow["cases"][diagnostic.CASE_NAMES[-1]]["timing"][
        "passes_frozen_speed_gate"
    ] = False
    assert diagnostic.decide_pallas_exact(slow)["promote"] is False
