"""Reproduce the execution-boundary v1 publication from immutable TPU artifacts.

Python stdlib only. --numbers-only skips profiles/download completeness;
default validates everything. --check byte-compares all derived outputs.
The prior run's lossless CSV and device-lane parser are deliberately reused.
No TPU invocation, source/default change, network request or raw-file mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.util
import json
import math
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
RAW = BASE / "execution_boundary/stream1_execution_boundary.json"
PRIOR_DIR = BASE.parent / "kaggle_layernorm_followup_v1"
CORPORA = {"legal_scrambles", "categorical_stress"}
COUNTS = dict(embedding=10, dense=18, dense_ln=18, layernorm=10,
              observations=4, controls=4, full_baselines=8, full=30, promotion=4)
TABLES = {"operator_summary.csv": ("embedding", "dense", "dense_ln", "layernorm"),
          "full_summary.csv": ("full_baselines", "full", "promotion"),
          "observation_summary.csv": ("observations",), "control_summary.csv": ("controls",)}


def import_helper(filename):
    spec = importlib.util.spec_from_file_location("prior_" + Path(filename).stem, PRIOR_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tables = import_helper("build_summary.py")
profiles = import_helper("profile_summary.py")
require, digest, load_strict = tables.require, tables.digest, tables.load_strict


def accepted(rows, batch):
    """Recompute frozen gate from raw metrics, not the permissive q.eligible flag."""
    groups = {}
    for row in rows:
        if row["batch"] != batch:
            continue
        group = groups.setdefault(row["id"], {})
        require(row["corpus"] not in group, "duplicate candidate/corpus/batch")
        group[row["corpus"]] = row
    result = []
    for key, group in groups.items():
        if set(group) != CORPORA:
            continue
        def passes(row):
            q = row.get("q", {}).get("unmasked", {})
            return (row["status"] == "ok" and not row.get("config", {}).get("control", True)
                    and row.get("timing_comparable") is True and q.get("finite") is True
                    and q.get("exact_fraction") == 1.0 and q.get("max_abs") == 0.0)
        if all(passes(row) for row in group.values()):
            result.append(key)
    return sorted(result)


def paired_summary(reference, candidate):
    require(bool(reference) and len(reference) == len(candidate), "unmatched timing samples")
    require(all(math.isfinite(x) and x > 0 for x in reference + candidate), "invalid timing sample")
    ratios = [a / b for a, b in zip(reference, candidate)]
    deltas = [a - b for a, b in zip(reference, candidate)]
    return dict(ratio_samples=ratios, ratio_median=statistics.median(ratios),
                ratio_min=min(ratios), ratio_max=max(ratios),
                delta_ms_samples=deltas, delta_ms_median=statistics.median(deltas),
                faster_rounds=sum(a > b for a, b in zip(reference, candidate)), rounds=len(ratios))


def check_timing(timing, count):
    values = timing["samples_ms"]
    require(len(values) == count and all(math.isfinite(v) and v > 0 for v in values), "bad samples")
    for key, expected in (("median_ms", statistics.median(values)), ("min_ms", min(values)), ("max_ms", max(values))):
        require(timing[key] == expected, "timing summary disagrees with samples")


def validate(report):
    prior = load_strict(PRIOR_DIR / "arithmetic_followup/stream1_layernorm_followup.json")
    require(report["status"] == "complete" and report["error_count"] == 0, "not this completed zero-error run")
    require(report["context"]["source_commit"] == "45062324d368f4849adb6d572d21d54f75854d79", "wrong source")
    for key in ("checkpoint_sha256", "original_source_sha256", "puzzle_sha256", "input_sha256", "seeds", "move_names"):
        require(report["context"][key] == prior["context"][key], "provenance mismatch: " + key)
    require(report["architecture"] == prior["architecture"], "architecture mismatch")
    runtime = report["context"]["runtime"]
    require(runtime["active_device_count"] == 1 and runtime["local_device_count"] == len(runtime["devices"]) == 8, "wrong device count")
    require(all(d["kind"] == "TPU v5 lite" for d in runtime["devices"]), "wrong target")
    require(all(runtime["versions"][k] == v for k, v in dict(jax="0.10.2", jaxlib="0.10.2", libtpu="0.0.42.1").items()), "runtime drift")
    for section, count in COUNTS.items():
        rows = report[section]
        require(len(rows) == count and all(r["status"] == "ok" for r in rows), "case count/status: " + section)
        require(len({(r["id"], r["corpus"], r["batch"]) for r in rows}) == len(rows), "duplicate row")
    require(len(report["timing_groups"]) == 12, "incomplete paired groups")
    covered = set()
    for group in report["timing_groups"]:
        require(group["status"] == "ok" and group["comparison_valid"] is True, "incomparable group")
        require(group["repeats"] == 12 and group["warmups"] == 5, "timing protocol drift")
        sections = [group["scope"]] + (["full_baselines"] if group["scope"] in ("full", "promotion") else [])
        rows = {r["id"]: r for s in sections for r in report[s]
                if r["corpus"] == group["corpus"] and r["batch"] == group["batch"]}
        require(set(rows) == set(group["cases"]), "timed row membership mismatch")
        order = group["execution_order"]
        require(len(order) == 12 and len(order[0]) == len(rows) and set(order[0]) == set(rows), "bad paired order")
        require(all(keys == (order[0] if i % 2 == 0 else order[0][::-1]) for i, keys in enumerate(order)), "nonalternating order")
        for key, row in rows.items():
            require(row["timing_comparable"] is True and row["timing"] == group["cases"][key], "unpaired row")
            check_timing(row["timing"], 12)
            check_timing(row["queued"], 5)
            require(row["queued"]["queue_depth"] == 8, "wrong queue depth")
            require(row["queued"]["samples_ms"] == [v / 8 for v in row["queued"]["batch_samples_ms"]], "wrong amortization")
            require(math.isclose(row["states_per_second"], row["batch"] * 1000 / row["timing"]["median_ms"], rel_tol=1e-12), "wrong throughput")
            covered.add((group["scope"], group["corpus"], key))
    good16, good32 = accepted(report["full"], 16384), accepted(report["promotion"], 32768)
    selected = report["promotion_decision"]["selected_for_larger_batch"]
    require(len(selected) <= 2 and set(selected) <= set(good16), "invalid promotion")
    require(set(r["id"] for r in report["promotion"]) == set(selected), "missing promotion execution")
    require(sorted(report["promotion_decision"]["exact_at_larger_batch"]) == good32, "32K claim mismatch")
    pairs = []
    for section, batch, good in (("full", 16384, good16), ("promotion", 32768, good32)):
        for row in report[section]:
            q = row["q"]["unmasked"]
            exact = q["finite"] and q["exact_fraction"] == 1.0
            require(row["exact_oracle_on_sample"] == exact, "raw exactness flag mismatch")
            require((row["mismatch_witnesses"]["mismatch_count"] == 0) == exact, "witness mismatch")
            for field, baseline_id in (("eligible_speedup", "original_runtime"), ("eligible_speedup_vs_typed", "typed_runtime")):
                if row["id"] not in good:
                    require(row[field] is None, "ineligible speedup published")
                    continue
                baseline = next(b for b in report["full_baselines"] if (b["id"], b["corpus"], b["batch"]) == (baseline_id, row["corpus"], batch))
                ratio = baseline["timing"]["median_ms"] / row["timing"]["median_ms"]
                require(math.isclose(row[field], ratio, rel_tol=1e-12), "speedup ratio mismatch")
                pairs.append(dict(section=section, id=row["id"], corpus=row["corpus"], batch=batch,
                                  baseline=baseline_id, ratio_of_medians=ratio,
                                  **paired_summary(baseline["timing"]["samples_ms"], row["timing"]["samples_ms"])))
    return dict(accepted_16K=good16, confirmed_32K=good32, paired_eligible_comparisons=pairs,
                matched_prior_context_fields=["checkpoint_sha256", "original_source_sha256", "puzzle_sha256", "input_sha256", "seeds", "move_names"],
                provenance_limit="Compares recorded runtime hashes, does not rehash external dataset assets locally.")


def profile_or_rejection(events):
    # New tiled embedding traces contain while containers on the SAME XLA Ops
    # lane as their body instructions. Drop only proven containers; all leaves
    # and the enclosing module stay subject to the old strict interval checks.
    pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
            and e.get("args", {}).get("name") == "/device:TPU:0"}
    lanes = {(e["pid"], e["tid"]) for e in events if e.get("ph") == "M" and e.get("name") == "thread_name"
             and e["pid"] in pids and e.get("args", {}).get("name") == "XLA Ops"}
    ops = [e for e in events if e.get("ph") == "X" and (e.get("pid"), e.get("tid")) in lanes]
    excluded = []
    for op in ops:
        if op["name"] != "while" and not op["name"].startswith("while."):
            continue
        if any(child is not op and op["ts"] - .001 <= child["ts"]
               and child["ts"] + child["dur"] <= op["ts"] + op["dur"] + .001 for child in ops):
            excluded.append(op)
    excluded_ids = {id(e) for e in excluded}
    leaves = [e for e in events if id(e) not in excluded_ids]
    try:
        result = profiles.summarize_events(leaves)
        result["excluded_loop_containers"] = dict(event_count=len(excluded),
            names=dict(Counter(e["name"] for e in excluded)),
            inclusive_sum_ms_not_added=sum(e["dur"] for e in excluded) / 1000,
            rule="Only while/while.N spans containing body operations on TPU0 XLA Ops; no host spans or crossing leaf overlaps accepted.")
        result["original_trace_event_count"] = len(events)
        return dict(status="verified", **result)
    except ValueError as exc:
        return dict(status="rejected", error=str(exc))


def build_profiles(report):
    expected = {Path(r["profile"]["directory"]).name: r for r in report["full_baselines"] + report["full"] if "profile" in r}
    paths = sorted((BASE / "execution_boundary/profiles").rglob("*.trace.json.gz"))
    require(len(expected) == len(paths) == 17, "missing profile artifacts")
    result, rejected, seen = [], [], set()
    for path in paths:
        key = path.parents[3].name
        require(key in expected and key not in seen, "unknown/duplicate profile")
        seen.add(key)
        row = expected[key]
        profile = profile_or_rejection(json.loads(gzip.decompress(path.read_bytes()))["traceEvents"])
        if profile["status"] == "rejected":
            rejected.append(dict(profile, id=row["id"], trace_path=path.relative_to(BASE).as_posix(), trace_sha256=digest(path)))
            continue
        config = row.get("config", {})
        require(profile["categories"]["pallas_dense"]["event_count"] == (60 if config.get("dense") == "late" else 0), "wrong Dense dispatch count")
        require(profile["categories"]["pallas_ln"]["event_count"] == (60 if config.get("norm") == "experimental" else 0), "wrong LN dispatch count")
        xplane = path.with_name(path.name.replace(".trace.json.gz", ".xplane.pb"))
        require(xplane.stat().st_size > 0, "empty XPlane")
        profile.update(id=row["id"], batch=row["batch"], corpus=row["corpus"], config=config,
                       quality_exact_vs_original=(None if row["id"] == "original_runtime" else row.get("exact_oracle_on_sample", row.get("quality_vs_original", {}).get("unmasked", {}).get("exact_fraction") == 1.0)),
                       synchronous_median_ms=row["timing"]["median_ms"], queued_amortized_median_ms=row["queued"]["median_ms"],
                       trace_path=path.relative_to(BASE).as_posix(), trace_sha256=digest(path),
                       xplane_path=xplane.relative_to(BASE).as_posix(), xplane_sha256=digest(xplane), xplane_parsed=False)
        result.append(profile)
    require(seen == set(expected), "incomplete profile set")
    return dict(source_json_sha256=digest(RAW), declared_profile_count=len(expected), profile_count=len(result), profiles=result,
                rejected_profiles=rejected,
                scope="Three legal16K calls per compiled case; TPU0 XLA Ops only. Modules and host spans are not added to operator totals. Chrome microseconds crosschecked against device picoseconds; failed checks retained as rejected profiles, never rescaled. XPlane retained, not decoded.",
                caution="Diagnostic profile, not paired timing, queued caller, hardware counters or real128chunk scan. Original oracle has no measured self-comparison.")


def verify_download():
    manifest = load_strict(BASE / "download_manifest.json")
    entries = manifest["files"]
    require(len(entries) == manifest["listed_output_count"] + 1, "incomplete download manifest")
    expected = set()
    for item in entries:
        path = (BASE / item["path"]).resolve()
        require(path.is_relative_to(BASE) and item["path"] not in expected, "invalid artifact path")
        expected.add(item["path"])
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], "artifact integrity mismatch: " + item["path"])
        if path.name.endswith(".compiled.txt"):
            require(path.read_bytes().startswith(b"HloModule"), "not compiled HLO")
        if path.name.endswith(".stablehlo.txt"):
            require(b"module" in path.read_bytes()[:300], "not StableHLO")
    actual = {p.relative_to(BASE).as_posix() for p in (BASE / "execution_boundary").rglob("*") if p.is_file()}
    actual.add("tpu-execution-boundary-ab.log")
    require(actual == expected, "local raw files differ from remote output inventory")
    return dict(files=len(entries), total_bytes=sum(i["bytes"] for i in entries), manifest_sha256=digest(BASE / "download_manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    names = [*TABLES, "summary.json"] + ([] if args.numbers_only else ["profile_summary.json", "profile_summary.csv"])
    before = {name: (BASE / name).read_bytes() for name in names} if args.check else None
    raw_hash = digest(RAW)
    report = load_strict(RAW)
    verification = validate(report)
    csv_results = {name: tables.write_table(BASE / name, report, sections) for name, sections in TABLES.items()}
    summary = dict(kind="mechanical_summary_not_new_measurement", source_json_sha256=raw_hash,
                   context=report["context"], architecture=report["architecture"], protocol=report["protocol"],
                   counts={s: dict(total=len(report[s]), statuses=dict(Counter(r["status"] for r in report[s]))) for s in COUNTS},
                   verification=verification, promotion_decision=report["promotion_decision"], tables=csv_results,
                   generator_sha256=digest(Path(__file__)), helper_sha256={f: digest(PRIOR_DIR / f) for f in ("build_summary.py", "profile_summary.py")},
                   caveats=["Raw q.eligible is finite-only, not acceptance.", "Exactness is on these measured corpora, not all possible inputs.", "Global minimizing topK is a proxy, not distributed beam/replay.", "Original baseline has no self-quality metric; absent values are intentional.", "Queued same-executable calls are not real128chunk scan."])
    (BASE / "summary.json").write_bytes((json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    if not args.numbers_only:
        download = verify_download()
        profile_summary = build_profiles(report)
        for name, value in profiles.render_outputs(profile_summary).items():
            (BASE / name).write_bytes(value.encode())
        print(json.dumps(dict(download=download, profiles=profile_summary["profile_count"])))
    require(digest(RAW) == raw_hash, "raw JSON mutated")
    if before is not None:
        require(all((BASE / name).read_bytes() == data for name, data in before.items()), "nondeterministic derived output")
    print(json.dumps(dict(status="verified", accepted_16K=verification["accepted_16K"],
                          confirmed_32K=verification["confirmed_32K"], tables=csv_results), indent=2))


if __name__ == "__main__":
    main()
