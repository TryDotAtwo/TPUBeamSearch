"""Publication must not promote an inexact, incomplete or incomparable result."""
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "test_results/kaggle_execution_boundary_v1/analyze.py"
spec = importlib.util.spec_from_file_location("execution_boundary_analysis", PATH)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def rows():
    return [dict(id="candidate", corpus=corpus, batch=16384, status="ok",
                 config=dict(control=False), timing_comparable=True,
                 q=dict(unmasked=dict(finite=True, exact_fraction=1.0, max_abs=0.0)))
            for corpus in ("legal_scrambles", "categorical_stress")]


def test_gate_requires_each_corpus_at_the_same_batch_and_excludes_controls():
    candidates = rows()
    assert analysis.accepted(candidates, 16384) == ["candidate"]
    assert analysis.accepted(candidates[:1], 16384) == []
    assert analysis.accepted(candidates, 32768) == []
    candidates[1]["batch"] = 32768
    assert analysis.accepted(candidates, 16384) == []
    candidates = rows()
    for row in candidates:
        row["config"]["control"] = True
    assert analysis.accepted(candidates, 16384) == []


@pytest.mark.parametrize("change", ["inexact", "nonfinite", "unpaired", "failed"])
def test_gate_rejects_one_bad_corpus_even_with_eligible_flag(change):
    candidates = rows()
    bad = candidates[1]
    bad["q"]["unmasked"]["eligible"] = True
    if change == "inexact":
        bad["q"]["unmasked"].update(exact_fraction=0.999999, max_abs=0.0001)
    elif change == "nonfinite":
        bad["q"]["unmasked"]["finite"] = False
    elif change == "unpaired":
        bad["timing_comparable"] = False
    else:
        bad["status"] = "error"
    assert analysis.accepted(candidates, 16384) == []


def test_duplicate_corpus_is_not_two_corpus_confirmation():
    candidates = rows()
    candidates.append(candidates[0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        analysis.accepted(candidates, 16384)


def test_paired_summary_uses_matched_samples_not_ratio_of_averages():
    result = analysis.paired_summary([8.0, 12.0, 20.0], [4.0, 8.0, 25.0])
    assert result["ratio_samples"] == [2.0, 1.5, 0.8]
    assert result["ratio_median"] == 1.5
    assert result["faster_rounds"] == 2
    assert result["delta_ms_samples"] == [4.0, 4.0, -5.0]


@pytest.mark.parametrize("reference,candidate", [([], []), ([1], [1, 2]), ([1], [0]), ([1], [float("nan")])])
def test_invalid_timing_samples_cannot_generate_speedup(reference, candidate):
    with pytest.raises(ValueError):
        analysis.paired_summary(reference, candidate)


def test_inconsistent_profile_is_reported_not_silently_rescaled():
    events = [
        dict(ph="M", pid=1, name="process_name", args=dict(name="/device:TPU:0")),
        dict(ph="M", pid=1, tid=2, name="thread_name", args=dict(name="XLA Modules")),
        dict(ph="M", pid=1, tid=3, name="thread_name", args=dict(name="XLA Ops")),
    ]
    for i in range(3):
        events.extend([
            dict(ph="X", pid=1, tid=2, name="module", ts=i * 2000, dur=1000,
                 args=dict(device_duration_ps="1000000000")),
            dict(ph="X", pid=1, tid=3, name="gather", ts=i * 2000, dur=500),
        ])
    assert analysis.profile_or_rejection(events)["status"] == "verified"
    events[3]["dur"] = 1005.957
    result = analysis.profile_or_rejection(events)
    assert result["status"] == "rejected"
    assert "picosecond" in result["error"]
    assert "module_median_ms" not in result
    assert events[3]["dur"] == 1005.957


def test_loop_container_and_its_children_are_not_counted_twice():
    events = [
        dict(ph="M", pid=1, name="process_name", args=dict(name="/device:TPU:0")),
        dict(ph="M", pid=1, tid=2, name="thread_name", args=dict(name="XLA Modules")),
        dict(ph="M", pid=1, tid=3, name="thread_name", args=dict(name="XLA Ops")),
    ]
    for i in range(3):
        start = i * 2000
        events.extend([
            dict(ph="X", pid=1, tid=2, name="module", ts=start, dur=1000,
                 args=dict(device_duration_ps="1000000000")),
            dict(ph="X", pid=1, tid=3, name="while.3", ts=start, dur=900),
            dict(ph="X", pid=1, tid=3, name="gather", ts=start, dur=400),
            dict(ph="X", pid=1, tid=3, name="copy", ts=start + 500, dur=300),
        ])
    result = analysis.profile_or_rejection(events)
    assert result["status"] == "verified"
    assert result["device_op_count"] == 6
    assert result["device_ops_ms_per_forward"] == pytest.approx(0.7)
    assert result["excluded_loop_containers"]["event_count"] == 3
    assert len(events) == 15
    # This is a crossing overlap, not containment; it must remain rejected.
    events[5]["dur"] = 600
    assert analysis.profile_or_rejection(events)["status"] == "rejected"
