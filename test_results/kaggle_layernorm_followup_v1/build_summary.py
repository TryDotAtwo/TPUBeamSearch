"""Regenerate lossless mechanical tables for the completed follow-up v1 run.

Run from any directory with Python 3.10+; only the six named derived artifacts
beside this script are written. No JAX, TPU execution, network or git is used.
CSV column names retain raw JSON paths. Arrays are compact JSON, booleans and
numbers use JSON serialization without rounding, null is ``null``, and an
absent path is an empty cell. Strings are ordinary CSV strings. The large
traceback field is left in the immutable source JSON, referenced by section
and zero-based source_index. No measured or accepted status is invented.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "arithmetic_followup" / "stream1_layernorm_followup.json"
PRIOR = BASE.parent / "kaggle_layernorm_arithmetic_v1" / "arithmetic_ab" / "stream1_layernorm_arithmetic.json"
EXPECTED = {"synthetic": 56, "operators": 36, "screen": 14, "controls": 2,
            "full_baselines": 6, "full": 14, "promotion": 0}
TABLES = {"full_summary.csv": ("full_baselines", "full"),
          "operator_summary.csv": ("operators",),
          "synthetic_summary.csv": ("synthetic",),
          "screen_summary.csv": ("screen",),
          "control_summary.csv": ("controls",)}
CORPORA = {"legal_scrambles", "categorical_stress"}
MISSING = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_strict(path):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON constant {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def scalar(value):
    return value if isinstance(value, str) else dumps(value)


def flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        path = f"{prefix}.{key}" if prefix else key
        if path == "traceback":
            continue
        if isinstance(value, dict) and value:
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def resolve(row, path):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def validate(report, prior):
    require(report["status"] == prior["status"] == "complete", "Both runs must be complete")
    for section, count in EXPECTED.items():
        require(len(report[section]) == count, f"Unexpected count: {section}")
        require(all(row["status"] in ("ok", "error") for row in report[section]),
                f"Nonterminal row in {section}")
    expected_errors = {"synthetic": 14, "operators": 10, "screen": 0,
                       "controls": 0, "full_baselines": 0, "full": 0, "promotion": 0}
    errors = []
    for section, count in expected_errors.items():
        failed = [(index, row) for index, row in enumerate(report[section]) if row["status"] == "error"]
        require(len(failed) == count, f"Unexpected errors: {section}")
        for index, row in failed:
            require(row["error_type"] == "MosaicError", "Unexpected failure type")
            require("failed to compile TPU kernel" in row["error"], "Failure is not a compile rejection")
            require("timing" not in row and "queued" not in row, "Compile failure has execution timing")
            errors.append({"source_section": section, "source_index": index,
                           "id": row["id"], "error_type": row["error_type"],
                           "error": row["error"], "traceback_in_source": bool(row.get("traceback"))})
    require(len(errors) == report["error_count"] == 24, "Expected exactly 24 compile failures")
    require(report["architecture"] == prior["architecture"], "Architecture mismatch")
    identities = {}
    for field in ("checkpoint_sha256", "original_source_sha256", "puzzle_sha256", "input_sha256"):
        current = report["context"][field]
        previous = prior["context"][field]
        require(current == previous, f"Prior recorded identity mismatch: {field}")
        values = current.values() if isinstance(current, dict) else (current,)
        require(all(isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value) for value in values),
                f"Invalid SHA256: {field}")
        identities[field] = current
    for field in ("seeds", "move_names", "scramble_depth_counts"):
        require(report["context"][field] == prior["context"][field], f"Prior mismatch: {field}")
    require(set(report["context"]["input_sha256"]) == CORPORA, "Unexpected corpora")
    for name in CORPORA:
        require(report["corpus_statistics"][name]["input_sha256"] == identities["input_sha256"][name],
                f"Corpus-statistics hash mismatch: {name}")
        require(report["corpus_statistics"][name]["batch"] == 16384, "Unexpected corpus batch")
    protocol = report["protocol"]
    for key, value in {"screen_batch": 4096, "full_batch": 16384, "promotion_batch": 32768,
                       "warmups": 5, "repeats": 12, "queue_depth": 8, "queue_repeats": 5,
                       "no_backtrack": False}.items():
        require(protocol[key] == value, f"Unexpected protocol value: {key}")
    require(len(report["timing_groups"]) == 8, "Expected eight timing groups")
    require(all(g["status"] == "ok" and g["comparison_valid"] for g in report["timing_groups"]),
            "Expected valid paired timing groups")
    for section in ("synthetic", "operators"):
        for row in report[section]:
            if row["status"] == "ok":
                require(row["comparison"]["finite"] is True, f"Nonfinite {section}/{row['id']}")
    for row in report["full"]:
        q = row["q"]["unmasked"]
        exact = q["finite"] is True and q["exact_fraction"] == 1.0
        require(row["exact_oracle_on_sample"] == exact, "Incorrect recorded exact gate")
        require(exact == row["config"]["control"], "Only graph controls should be exact in this run")
        require(row["exact_across_corpora"] == exact, "Incorrect across-corpus exact gate")
        require(row["eligible_speedup"] is None and row["eligible_speedup_vs_typed"] is None,
                "This run must not publish eligible candidate speedups")
        require(row["timing_comparable"] is True, "Unexpected full timing comparability")
        require(q["finite"] is True, "Unexpected nonfinite full Q")
        for mode in ("unmasked", "inverse_mask_diagnostic"):
            scores = row["q"][mode]
            require(scores["score_direction"] == "minimize", "Wrong score direction")
            require(scores["selected_invalid_count"] == scores["all_masked_rows"] == 0,
                    "Unexpected invalid selection or all-masked row")
    for config in report["configurations"]:
        rows = [row for row in report["full"] if row["id"] == config["id"]]
        require(len(rows) == 2 and {r["corpus"] for r in rows} == CORPORA,
                f"Missing full corpus: {config['id']}")
    for row in report["full_baselines"]:
        if row["id"] != "original_runtime":
            q = row["quality_vs_original"]["unmasked"]
            require(q["finite"] is True and q["exact_fraction"] == 1.0,
                    "Typed/captured baseline must match original")
    for row in report["controls"]:
        require(row["same_suffix"]["unmasked"]["exact_fraction"] == 1.0,
                "Same-suffix zero replacement is not exact")
    require(report["promotion_decision"]["selected_for_larger_batch"] == [], "Unexpected promotion")
    require(report["promotion_decision"]["exact_at_larger_batch"] == [], "Unexpected 32K confirmation")
    return identities, errors


def write_table(path, report, sections):
    rows = []
    for section in sections:
        for index, raw in enumerate(report[section]):
            rows.append({"source_section": section, "source_index": index, **flatten(raw)})
    preferred = ["source_section", "source_index", "id", "corpus", "batch", "status",
                 "config.control", "exact_oracle_on_sample", "exact_across_corpora",
                 "timing_comparable", "timing_label", "eligible_speedup", "eligible_speedup_vs_typed"]
    present = {key for row in rows for key in row}
    columns = [key for key in preferred if key in present]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    require(len(columns) == len(set(columns)), "Duplicate CSV column")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: scalar(value) for key, value in row.items()} for row in rows)
    return verify_table(path, report, sections)


def verify_table(path, report, sections):
    """Resolve CSV paths directly against raw rows, independently of flatten()."""
    checked, numeric, arrays = 0, 0, 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    expected_rows = [(section, index) for section in sections for index in range(len(report[section]))]
    require(len(rows) == len(expected_rows), f"CSV row count mismatch: {path.name}")
    for row, (section, index) in zip(rows, expected_rows):
        require(row["source_section"] == section and int(row["source_index"]) == index,
                f"CSV source identity mismatch: {path.name}")
        raw = report[section][index]
        for key in columns:
            if key in ("source_section", "source_index"):
                continue
            expected = resolve(raw, key)
            cell = row[key]
            if expected is MISSING:
                require(cell == "", f"Unexpected CSV value: {key}")
                continue
            if isinstance(expected, str):
                require(cell == expected, f"CSV string mismatch: {key}")
            else:
                value = json.loads(cell)
                require(value == expected and type(value) is type(expected), f"CSV value mismatch: {key}")
                if isinstance(expected, float):
                    require(math.isfinite(expected) and value.hex() == expected.hex(), f"Rounded float: {key}")
                numeric += isinstance(expected, (int, float)) and not isinstance(expected, bool)
                arrays += isinstance(expected, list)
            checked += 1
    return {"rows": len(rows), "columns": len(columns), "raw_derived_cells_verified": checked,
            "numeric_cells_verified": numeric, "array_cells_verified": arrays, "sha256": digest(path)}


def main():
    before = {"source": digest(SOURCE), "prior": digest(PRIOR)}
    report, prior = load_strict(SOURCE), load_strict(PRIOR)
    identities, errors = validate(report, prior)
    tables = {filename: write_table(BASE / filename, report, sections) for filename, sections in TABLES.items()}
    counts = {section: {"total": len(report[section]),
                        "status_counts": dict(Counter(row["status"] for row in report[section]))}
              for section in EXPECTED}
    full = []
    for section in ("full_baselines", "full"):
        for index, row in enumerate(report[section]):
            keys = ("id", "corpus", "batch", "status", "config", "q", "quality_vs_original",
                    "typed_comparison", "exact_oracle_on_sample", "exact_across_corpora",
                    "eligible_speedup", "eligible_speedup_vs_typed", "timing_comparable",
                    "timing_label", "timing", "queued", "compilation", "states_per_second")
            full.append({"source_section": section, "source_index": index,
                         **{key: row[key] for key in keys if key in row}})
    summary = {
        "schema_version": 1,
        "kind": "mechanical_summary_not_new_measurement",
        "status": report["status"],
        "source_json": SOURCE.relative_to(BASE).as_posix(),
        "source_json_sha256": before["source"],
        "prior_json": "../" + PRIOR.relative_to(BASE.parent).as_posix(),
        "prior_json_sha256": before["prior"],
        "generator": "build_summary.py", "generator_sha256": digest(Path(__file__)),
        "regeneration_command": "python test_results/kaggle_layernorm_followup_v1/build_summary.py",
        "csv_encoding": {"charset": "UTF-8", "separator": ",", "line_ending": "LF",
                         "numbers": "JSON round-trip representation, no decimal rounding",
                         "booleans": "true/false", "arrays": "compact JSON", "null": "null",
                         "missing_field": "empty cell", "strings": "ordinary CSV strings",
                         "field_paths": "original JSON dotted paths; traceback omitted only",
                         "source_index": "zero-based row index within source_section"},
        "provenance": {"recorded_prior_identity_matches": identities,
                       "architecture_matches_prior": True,
                       "seeds_move_order_depth_counts_match_prior": True,
                       "followup_source_commit": report["context"]["source_commit"],
                       "prior_source_commit": prior["context"]["source_commit"],
                       "basis": "Recorded checkpoint, original model source, puzzle and input hashes match prior JSON; dataset assets are not rehashed by this transformation."},
        "context": report["context"], "architecture": report["architecture"],
        "protocol": report["protocol"], "corpus_statistics": report["corpus_statistics"],
        "counts": counts, "compile_error_count": len(errors), "compile_errors": errors,
        "promotion_decision": report["promotion_decision"],
        "accepted_noncontrol_full_configs": [], "larger_batch_measured": False,
        "tables": tables, "full_baselines_and_candidates": full,
        "cautions": [
            "Raw q.eligible means finite only; acceptance requires finite elementwise-exact original Q on both corpora.",
            "No non-control full candidate qualified; no eligible speedup or 32K measurement is reported.",
            "Original-runtime baseline has no recorded self-comparison; empty metric cells are intentional.",
            "Matching aggregate errors do not establish pairwise tensor equality or a causal mechanism.",
            "Full candidates retain JAX embedding/input/head; this is not an all-Pallas model.",
            "Stable minimizing global topK diagnostics do not establish distributed-beam or replay validity.",
            "Inverse masking is diagnostic only; stress last_move=-1 makes its mask all-valid.",
            "Queued calls reuse the same executable; they are neither one-call latency nor a real chunk scan.",
            "Profiles and timings remain diagnostics when the exact-Q gate fails.",
        ],
        "verification": {"expected_counts_asserted": EXPECTED, "compile_failures_asserted": 24,
                         "csv_cells_reloaded_and_compared_to_raw_paths": True,
                         "input_json_hashes_unchanged": True},
    }
    require(before == {"source": digest(SOURCE), "prior": digest(PRIOR)}, "An input JSON changed")
    target = BASE / "summary.json"
    target.write_bytes((json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    require(load_strict(target) == summary, "summary.json round-trip mismatch")
    print(json.dumps({"status": "verified", "tables": tables,
                      "source_json_sha256": before["source"], "prior_json_sha256": before["prior"],
                      "inputs_unchanged": True, "compile_errors": len(errors)}, indent=2))


def self_test():
    """Regression check for platform-independent LF bytes and determinism."""
    outputs = [BASE / name for name in (*TABLES, "summary.json")]
    for path in outputs:
        data = path.read_bytes()
        require(b"\r" not in data, f"{path.name} must use LF-only bytes: {data.count(bytes([13]))} CR bytes")
    main()
    first = {path.name: path.read_bytes() for path in outputs}
    main()
    for path in outputs:
        data = path.read_bytes()
        require(b"\r" not in data, f"{path.name} regenerated with CR bytes")
        require(data == first[path.name], f"{path.name} regeneration is not byte deterministic")
    print("SELF-TEST PASS: six LF-only outputs, deterministic regeneration, raw-cell verification passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        require(not sys.argv[1:], "Usage: build_summary.py [--self-test]")
        main()
