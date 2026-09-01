"""Validate and summarize exact-inference-frontier TPU profile artifacts."""

from __future__ import annotations

from collections import Counter
import importlib.util
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
PRIOR_PROFILE_HELPER = BASE.parent / "kaggle_layernorm_followup_v1" / "profile_summary.py"


def _load_profile_helper():
    spec = importlib.util.spec_from_file_location("frontier_profile_helper", PRIOR_PROFILE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load profile helper: {PRIOR_PROFILE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROFILE_HELPER = _load_profile_helper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _collapse_composed_modules(events: list[dict], device_pid: int) -> tuple[list[dict], dict]:
    """Represent three prefix+head dispatch pairs as three composed forwards."""
    module_lanes = {
        (event["pid"], event["tid"])
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "thread_name"
        and event.get("pid") == device_pid
        and event.get("args", {}).get("name") == "XLA Modules"
    }
    _require(len(module_lanes) == 1, "expected exactly one TPU0 XLA Modules lane")
    module_lane = next(iter(module_lanes))
    modules = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X" and (event.get("pid"), event.get("tid")) == module_lane
        ),
        key=lambda event: event["ts"],
    )
    if len(modules) == 3:
        return events, {
            "compiled_module_count": 3,
            "dispatches_per_forward": 1,
            "inter_dispatch_gap_ms_per_forward": 0.0,
        }
    _require(len(modules) == 6, "expected three single-dispatch calls or three two-dispatch calls")
    names = [module["name"] for module in modules]
    _require(names == names[:2] * 3, "composed module sequence is not three stable prefix/head pairs")

    synthetic_modules = []
    gaps_ms = []
    for left, right in zip(modules[::2], modules[1::2]):
        gap_us = right["ts"] - (left["ts"] + left["dur"])
        _require(gap_us >= -0.001, "composed prefix/head module executions overlap")
        end_us = right["ts"] + right["dur"]
        duration_us = end_us - left["ts"]
        gaps_ms.append(gap_us / 1000)
        synthetic_modules.append(
            {
                "ph": "X",
                "pid": device_pid,
                "tid": module_lane[1],
                "name": "composed_runner",
                "ts": left["ts"],
                "dur": duration_us,
                "args": {
                    "device_duration_ps": str(round(duration_us * 1_000_000)),
                    "run_id": f"{left.get('args', {}).get('run_id')}/{right.get('args', {}).get('run_id')}",
                },
            }
        )
    prepared = [
        event
        for event in events
        if not (event.get("ph") == "X" and (event.get("pid"), event.get("tid")) == module_lane)
    ]
    prepared.extend(synthetic_modules)
    return prepared, {
        "compiled_module_count": 6,
        "dispatches_per_forward": 2,
        "inter_dispatch_gap_ms_per_forward": sum(gaps_ms) / len(gaps_ms),
        "compiled_module_names": names[:2],
    }


def validate_frontier_report(report: dict) -> dict:
    """Recompute the frozen winner gate from the immutable benchmark JSON."""
    _require(report.get("status") == "complete", "frontier report is not complete")
    _require(
        report.get("context", {}).get("source_commit")
        == "fc5c87ae5c49c0a92d4ccd634831e8980a7f44e8",
        "unexpected source commit",
    )
    runtime = report["context"]["runtime"]
    devices = runtime["devices"]
    _require(
        runtime["active_device_count"] == runtime["local_device_count"] == len(devices) == 8,
        "benchmark did not use eight active devices",
    )
    _require(all(device["kind"] == "TPU v5 lite" for device in devices), "unexpected TPU kind")
    _require(
        len(report["input_scopes"]) == 4
        and all(len(scope["shard_input_sha256"]) == 8 for scope in report["input_scopes"]),
        "incomplete input shard provenance",
    )
    expected_counts = {
        "head_measurements": 86,
        "prefix_measurements": 8,
        "full_configurations": 19,
        "full_measurements": 38,
        "timing_groups": 12,
        "profiles": 16,
    }
    for name, expected in expected_counts.items():
        _require(len(report[name]) == expected, f"unexpected {name} count")

    for group in report["timing_groups"]:
        _require(group["status"] == "ok" and group["comparison_valid"] is True, "invalid timing group")
        _require(group["warmups"] == 5 and group["repeats"] == 12, "timing protocol drift")
        order = group["execution_order"]
        _require(len(order) == 12, "incomplete timing execution order")
        _require(
            all(items == (order[0] if index % 2 == 0 else order[0][::-1]) for index, items in enumerate(order)),
            "timing order is not alternating",
        )

    decision = report["confirmation_decision"]
    selected_id = decision["selected_id"]
    accepted_id = decision["accepted_control_id"]
    _require(
        decision["confirmed"] is True
        and decision["improvement_achieved"] is True
        and decision["screen_selected_id"] == selected_id
        and report["screen_decision"]["selected_id"] == selected_id,
        "winner was not independently confirmed",
    )

    rows = {
        (row["section"], row["corpus"], row["id"]): row
        for row in report["full_measurements"]
    }
    winner_comparisons = []
    for section in ("screen", "confirmation"):
        for corpus in ("legal_scrambles", "categorical_stress"):
            winner = rows[(section, corpus, selected_id)]
            control = rows[(section, corpus, accepted_id)]
            quality = winner["comparison_vs_original"]
            witnesses = winner["mismatch_witnesses"]
            _require(
                winner["status"] == "ok"
                and quality["finite"] is True
                and quality["exact_fraction"] == 1.0
                and quality["max_abs"] == quality["mean_abs"] == quality["rmse"] == 0.0
                and witnesses["mismatch_count"] == 0
                and witnesses["nonfinite_reference"] == witnesses["nonfinite_candidate"] == 0,
                "winner is not elementwise exact",
            )
            _require(winner["output_sha256"] == control["output_sha256"], "winner output hash differs")
            winner_ms = winner["timing"]["median_ms"]
            control_ms = control["timing"]["median_ms"]
            _require(
                math.isfinite(winner_ms) and math.isfinite(control_ms) and winner_ms < control_ms,
                "winner does not beat accepted control",
            )
            winner_comparisons.append(
                {
                    "section": section,
                    "corpus": corpus,
                    "winner_ms": winner_ms,
                    "control_ms": control_ms,
                    "speedup": control_ms / winner_ms,
                    "states_per_second": winner["states_per_second"],
                }
            )

    rejection_ids = sorted(error["id"] for error in report["compile_errors"])
    _require(rejection_ids == ["prefix_bm16384", "prefix_bm8192"], "unexpected compile rejection set")
    _require(
        all("CompileTimeScopedVmemOom" in error["error"] for error in report["compile_errors"]),
        "compile rejection is not scoped VMEM OOM",
    )
    return {
        "selected_id": selected_id,
        "accepted_control_id": accepted_id,
        "winner_comparisons": winner_comparisons,
        "compile_rejection_ids": rejection_ids,
    }


def summarize_profile_events(events: list[dict]) -> dict:
    """Summarize TPU0 events after removing proven inclusive while spans."""
    device_pids = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "/device:TPU:0"
    }
    lanes = {
        (event["pid"], event["tid"])
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "thread_name"
        and event.get("pid") in device_pids
        and event.get("args", {}).get("name") == "XLA Ops"
    }
    scoped_events = [event for event in events if event.get("pid") in device_pids]
    operations = [
        event
        for event in scoped_events
        if event.get("ph") == "X" and (event.get("pid"), event.get("tid")) in lanes
    ]
    excluded = []
    for operation in operations:
        if operation["name"] != "while" and not operation["name"].startswith("while."):
            continue
        if any(
            child is not operation
            and operation["ts"] - 0.001 <= child["ts"]
            and child["ts"] + child["dur"] <= operation["ts"] + operation["dur"] + 0.001
            for child in operations
        ):
            excluded.append(operation)
    excluded_ids = {id(event) for event in excluded}
    leaves = [event for event in scoped_events if id(event) not in excluded_ids]
    try:
        _require(len(device_pids) == 1, "expected exactly one TPU0 process")
        prepared, execution = _collapse_composed_modules(leaves, next(iter(device_pids)))
        result = _PROFILE_HELPER.summarize_events(prepared)
    except ValueError as exc:
        return {"status": "rejected", "error": str(exc)}
    result.update(execution)
    result["status"] = "verified"
    result["excluded_loop_containers"] = {
        "event_count": len(excluded),
        "names": dict(sorted(Counter(event["name"] for event in excluded).items())),
        "inclusive_sum_ms_not_added": sum(event["dur"] for event in excluded) / 1000,
    }
    result["original_trace_event_count"] = len(events)
    result["tpu0_trace_event_count"] = len(scoped_events)
    return result
