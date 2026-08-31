"""Reproduce device-only attribution; Python stdlib, no TPU/JAX dependency.

Run without arguments to regenerate profile_summary.json/.csv beside this file.
--check validates raw traces and byte-compares existing generated outputs.
--self-test exercises host exclusion, device overlap rejection, and time units.
Raw benchmark JSON, trace gzip and XPlane protobuf files are never modified.
"""

from __future__ import annotations

import copy
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import unittest


def summarize_events(events):
    processes = {e["pid"]: e["args"]["name"] for e in events
                 if e.get("ph") == "M" and e.get("name") == "process_name"}
    device_pids = [pid for pid, name in processes.items() if name.startswith("/device:")]
    if len(device_pids) != 1 or processes[device_pids[0]] != "/device:TPU:0":
        raise ValueError("expected exactly one traced device: /device:TPU:0")
    pid = device_pids[0]
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in events
               if e.get("ph") == "M" and e.get("name") == "thread_name"}
    lanes = {}
    for name in ("XLA Modules", "XLA Ops"):
        keys = [key for key, value in threads.items() if key[0] == pid and value == name]
        if len(keys) != 1:
            raise ValueError(f"expected exactly one TPU0 {name} lane")
        lanes[name] = sorted([e for e in events if e.get("ph") == "X"
                              and (e.get("pid"), e.get("tid")) == keys[0]],
                             key=lambda e: e["ts"])
    modules, ops = lanes["XLA Modules"], lanes["XLA Ops"]
    if len(modules) != 3 or not ops:
        raise ValueError("expected three module executions and populated device ops")
    tolerance_us = 0.001  # 1 ns; Chrome JSON vs device picosecond timestamp rounding.
    for name, lane in lanes.items():
        for e in lane:
            if not all(math.isfinite(e[k]) for k in ("ts", "dur")) or e["dur"] < 0:
                raise ValueError(f"invalid timestamp/duration in {name}")
        for left, right in zip(lane, lane[1:]):
            if left["ts"] + left["dur"] > right["ts"] + tolerance_us:
                raise ValueError(f"overlap in {name}: {left['name']} / {right['name']}")
    for module in modules:
        ps = float(module["args"]["device_duration_ps"])
        if abs(module["dur"] - ps / 1_000_000) > tolerance_us:
            raise ValueError("Chrome microsecond duration disagrees with device picosecond duration")
    members = [[] for _ in modules]
    for op in ops:
        owners = [i for i, m in enumerate(modules)
                  if m["ts"] - tolerance_us <= op["ts"]
                  and op["ts"] + op["dur"] <= m["ts"] + m["dur"] + tolerance_us]
        if len(owners) != 1:
            raise ValueError(f"device op has no unique module owner: {op['name']}")
        members[owners[0]].append(op)
    categories = {name: {"event_count": 0, "sum_ms": 0.0} for name in
                  ("gather", "reshape", "pallas_dense", "pallas_ln", "other")}
    by_name = {}
    for op in ops:
        name = op["name"]
        category = ("pallas_ln" if name.startswith("stream1_ln_") else
                    "pallas_dense" if name.startswith("stream1_dense_") else
                    "gather" if name.startswith("gather") else
                    "reshape" if name.startswith("reshape") else "other")
        info = by_name.setdefault(name, {"category": category, "event_count": 0,
                                        "samples_ms": [], "sum_ms": 0.0})
        ms = op["dur"] / 1000
        info["event_count"] += 1
        info["samples_ms"].append(ms)
        info["sum_ms"] += ms
        categories[category]["event_count"] += 1
        categories[category]["sum_ms"] += ms
    for info in list(categories.values()) + list(by_name.values()):
        info["count_per_forward"] = info["event_count"] / len(modules)
        info["ms_per_forward"] = info["sum_ms"] / len(modules)
    samples = [m["dur"] / 1000 for m in modules]
    op_samples = [sum(e["dur"] for e in group) / 1000 for group in members]
    if any(op_ms > module_ms + tolerance_us / 1000 for op_ms, module_ms in zip(op_samples, samples)):
        raise ValueError("device operation sum exceeds enclosing module duration")
    return {
        "device": processes[pid], "trace_event_count": len(events),
        "trace_phase_counts": dict(sorted(collections.Counter(e.get("ph", "missing") for e in events).items())),
        "module_count": len(modules), "module_samples_ms": samples,
        "module_median_ms": statistics.median(samples), "module_mean_ms": statistics.mean(samples),
        "module_min_ms": min(samples), "module_max_ms": max(samples),
        "modules": [{"name": m["name"], "ts_us": m["ts"], "duration_us": m["dur"],
                     "device_duration_ps": m["args"]["device_duration_ps"],
                     "run_id": m["args"].get("run_id"), "device_op_count": len(group),
                     "device_ops_sum_ms": op_ms, "unattributed_module_ms": ms - op_ms}
                    for m, group, op_ms, ms in zip(modules, members, op_samples, samples)],
        "device_op_count": len(ops), "device_ops_ms_per_forward": statistics.mean(op_samples),
        "categories": categories, "operations_by_name": dict(sorted(by_name.items())),
        "checks": {"three_module_calls": True, "single_device_TPU0": True,
                   "host_spans_excluded": True, "no_device_lane_overlap": True,
                   "each_device_op_has_one_module": True, "picosecond_unit_crosscheck": True},
    }


def build_summary(root):
    benchmark_path = root / "arithmetic_followup" / "stream1_layernorm_followup.json"
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    runtime = benchmark["context"]["runtime"]
    if runtime["active_device_count"] != 1 or runtime["local_device_count"] != 8:
        raise ValueError("benchmark runtime does not match scoped one-active/eight-visible run")
    if len(runtime["devices"]) != 8 or any(d["kind"] != "TPU v5 lite" for d in runtime["devices"]):
        raise ValueError("unexpected benchmark device inventory")
    expected = {}
    for row in benchmark["full_baselines"] + benchmark["full"]:
        if row["corpus"] == "legal_scrambles" and row["batch"] == 16384:
            if row.get("status") != "ok" or row.get("profile", {}).get("iterations") != 3:
                raise ValueError(f"missing successful three-call profile metadata: {row['id']}")
            expected[Path(row["profile"]["directory"]).name] = row
    paths = sorted((root / "arithmetic_followup" / "profiles").rglob("*.trace.json.gz"))
    if len(expected) != 10 or len(paths) != 10:
        raise ValueError("expected exactly ten JSON-declared profiles and ten trace gzip files")
    profiles, seen = [], set()
    for path in paths:
        key = path.parents[3].name
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate trace: {key}")
        seen.add(key)
        row = expected[key]
        raw = path.read_bytes()
        trace = json.loads(gzip.decompress(raw))
        profile = summarize_events(trace["traceEvents"])
        config = row.get("config", {})
        expected_ln = 20 if config and config["norm"] != "jax" else 0
        expected_dense = 20 if config.get("dense") == "late" else 0
        if profile["categories"]["pallas_ln"]["event_count"] != 3 * expected_ln:
            raise ValueError(f"LN trace count disagrees with benchmark configuration: {key}")
        if profile["categories"]["pallas_dense"]["event_count"] != 3 * expected_dense:
            raise ValueError(f"Dense trace count disagrees with benchmark configuration: {key}")
        if any(m["device_op_count"] != profile["modules"][0]["device_op_count"] for m in profile["modules"]):
            raise ValueError(f"device operator count changes between identical calls: {key}")
        xplane = path.with_name(path.name.replace(".trace.json.gz", ".xplane.pb"))
        xplane_bytes = xplane.read_bytes()
        if not xplane_bytes:
            raise ValueError(f"empty XPlane artifact: {xplane}")
        profile.update({"id": row["id"], "batch": row["batch"], "corpus": row["corpus"],
                        "config": config, "trace_path": path.relative_to(root).as_posix(),
                        "trace_sha256": hashlib.sha256(raw).hexdigest(), "trace_bytes": len(raw),
                        "xplane_path": xplane.relative_to(root).as_posix(),
                        "xplane_sha256": hashlib.sha256(xplane_bytes).hexdigest(),
                        "xplane_bytes": len(xplane_bytes), "xplane_parsed": False,
                        "synchronous_median_ms": row["timing"]["median_ms"],
                        "queued_amortized_median_ms": row["queued"]["median_ms"],
                        "quality_exact_vs_original": (True if row["id"] == "original_runtime" else
                            row.get("exact_oracle_on_sample", row.get("quality_vs_original", {}).get("unmasked", {}).get("exact_fraction") == 1.0))})
        # These reads are intentionally repeated to prove the summarizer is non-mutating.
        if path.read_bytes() != raw or xplane.read_bytes() != xplane_bytes:
            raise ValueError(f"raw profile changed during analysis: {key}")
        profiles.append(profile)
    if seen != set(expected) or benchmark_path.read_bytes() != benchmark_bytes:
        raise ValueError("incomplete profile set or benchmark JSON changed during analysis")
    return {
        "schema_version": 1,
        "benchmark_path": benchmark_path.relative_to(root).as_posix(),
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "source_commit": benchmark["context"]["source_commit"], "runtime": runtime,
        "profile_count": len(profiles),
        "scope": "Three synchronous diagnostic full16K legal calls per case, after timing; not queued execution or real chunked caller.",
        "time_units": "Chrome ts/dur microseconds; divide dur by1000 for milliseconds. Each module crosschecked against device_duration_ps/1e9 milliseconds within1ns.",
        "classification": {
            "lane": "Only ph=X on the unique /device:TPU:0 XLA Ops lane is summed; XLA Modules are separate enclosing measurements. Never add host, Async XLA Ops, module and op totals together.",
            "gather": "op name startswith gather; observed gather_fusion embedding gather",
            "reshape": "op name startswith reshape; includes all such device ops, not only embedding flatten",
            "pallas_dense": "op name startswith stream1_dense_; JAX Dense/fused operations remain other",
            "pallas_ln": "op name startswith stream1_ln_; JAX LN/fused operations remain other",
            "other": "all remaining XLA Ops names, retained individually under operations_by_name",
            "aggregation": "sum_ms is across all three calls; ms_per_forward=sum_ms/3, count_per_forward=event_count/3. Per-module counts and sums checked independently.",
        },
        "limitations": [
            "Failed-quality candidates remain diagnostic, irrespective of timing or profile availability.",
            "XPlane protobuf files are retained and hashed but not decoded; attribution uses real Chrome device trace events.",
            "No optional JAX/MLIR decode is required; embedded StableMosaic is pre-layout, not final machine IR.",
            "Profiler samples are a separate diagnostic run and are not interchangeable with twelve unprofiled timing samples.",
        ],
        "profiles": profiles,
    }


def render_outputs(summary):
    json_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rows = []
    for p in summary["profiles"]:
        row = {key: p[key] for key in ("id", "batch", "corpus", "quality_exact_vs_original", "module_count",
               "module_median_ms", "module_mean_ms", "module_min_ms", "module_max_ms", "device_op_count",
               "device_ops_ms_per_forward", "synchronous_median_ms", "queued_amortized_median_ms", "trace_path")}
        for i, sample in enumerate(p["module_samples_ms"], 1):
            row[f"module_sample_{i}_ms"] = sample
        for name, category in p["categories"].items():
            row[f"{name}_count_per_forward"] = category["count_per_forward"]
            row[f"{name}_ms_per_forward"] = category["ms_per_forward"]
        rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return {"profile_summary.json": json_text, "profile_summary.csv": stream.getvalue()}


class ProfileSummaryTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"ph": "M", "pid": 3, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 3, "tid": 2, "name": "thread_name", "args": {"name": "XLA Modules"}},
            {"ph": "M", "pid": 3, "tid": 3, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "M", "pid": 701, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "M", "pid": 701, "tid": 3, "name": "thread_name", "args": {"name": "XLA Ops"}},
        ]
        for i in range(3):
            start = i * 2000
            self.events.extend([
                {"ph": "X", "pid": 3, "tid": 2, "name": "jit_call(42)", "ts": start, "dur": 1000,
                 "args": {"device_duration_ps": "1000000000", "run_id": str(i)}},
                {"ph": "X", "pid": 3, "tid": 3, "name": "gather_fusion", "ts": start, "dur": 400},
                {"ph": "X", "pid": 3, "tid": 3, "name": "reshape.1", "ts": start + 400, "dur": 100},
                {"ph": "X", "pid": 3, "tid": 3, "name": "stream1_dense_linear.20", "ts": start + 500, "dur": 200},
                {"ph": "X", "pid": 3, "tid": 3, "name": "stream1_ln_experimental_hlo_mixed_fp32_where.20", "ts": start + 700, "dur": 250},
                {"ph": "X", "pid": 701, "tid": 3, "name": "nested_host_span", "ts": start, "dur": 999999},
            ])

    def test_three_device_calls_exclude_nested_host_spans_and_preserve_categories(self):
        result = summarize_events(self.events)
        self.assertEqual(result.get("module_samples_ms"), [1.0, 1.0, 1.0])
        self.assertEqual(result["device_op_count"], 12)
        self.assertAlmostEqual(result["device_ops_ms_per_forward"], 0.95)
        self.assertAlmostEqual(result["categories"]["pallas_ln"]["ms_per_forward"], 0.25)
        self.assertEqual(result["categories"]["pallas_dense"]["count_per_forward"], 1)

    def test_overlapping_device_operations_rejected(self):
        self.events.append({"ph": "X", "pid": 3, "tid": 3, "name": "nested", "ts": 20, "dur": 10})
        with self.assertRaisesRegex(ValueError, "overlap"):
            summarize_events(self.events)

    def test_duration_unit_mismatch_rejected(self):
        events = copy.deepcopy(self.events)
        next(e for e in events if e.get("name") == "jit_call(42)")["args"]["device_duration_ps"] = "1000000"
        with self.assertRaisesRegex(ValueError, "picosecond"):
            summarize_events(events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate raw inputs and compare generated outputs without writes")
    parser.add_argument("--self-test", action="store_true", help="run three focused stdlib tests without artifact writes")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    else:
        root = Path(__file__).resolve().parent
        summary = build_summary(root)
        for name, content in render_outputs(summary).items():
            output = root / name
            expected_bytes = content.encode("utf-8")
            if args.check:
                if not output.is_file() or output.read_bytes() != expected_bytes:
                    raise SystemExit(f"MISMATCH: {name}; regenerate using this script without --check")
            else:
                output.write_bytes(expected_bytes)
        print(f"{'CHECKED' if args.check else 'WROTE'}: {summary['profile_count']} profiles, "
              f"{sum(p['module_count'] for p in summary['profiles'])} TPU0 module calls; "
              "device-only nonoverlapping ops, picosecond crosscheck, raw bytes unchanged")
