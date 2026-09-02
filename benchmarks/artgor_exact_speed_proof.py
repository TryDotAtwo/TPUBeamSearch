"""Strict eight-TPU speed proof for exact Artgor full-Q inference."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.artgor_exact_notebook_validation import (
    _array_sha256,
    _dataset_path,
    _make_original_inference,
    _replicate,
    _tensor_comparison,
    checkpoint,
)
from benchmarks.layernorm_quality import load_puzzle, make_legal_scrambles
from benchmarks.stream1_layernorm_arithmetic import (
    runtime_inventory,
    sha256_file,
)
from tpu_beam_search.artgor_staged_beam import (
    ArtgorExactConfig,
    prepare_artgor_exact_beam_runtime,
)


TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 32_768
GLOBAL_BATCH = TARGET_DEVICE_COUNT * LOCAL_BATCH
WARMUPS = 3
REPEATS = 21
MIN_SPEEDUP = 1.5
BOOTSTRAP_SAMPLES = 20_000
RESULT_NAME = "artgor_exact_speed_proof.json"
CASE_DEFINITIONS = (
    ("legal_seed_42", "legal", 42),
    ("legal_seed_142", "legal", 142),
    ("legal_seed_242", "legal", 242),
    ("stress_seed_43", "stress", 43),
    ("stress_seed_143", "stress", 143),
    ("stress_seed_243", "stress", 243),
)
CASE_NAMES = tuple(name for name, _, _ in CASE_DEFINITIONS)


def _finite_positive(values, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain finite positive timings")
    return result


def _binomial_upper_tail(successes: int, trials: int) -> float:
    return sum(
        math.comb(trials, count) for count in range(successes, trials + 1)
    ) / (2**trials)


def paired_speed_statistics(
    *,
    baseline_s,
    candidate_s,
    orders,
    threshold: float = MIN_SPEEDUP,
    bootstrap_seed: int,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict:
    """Summarize paired AB/BA samples under the frozen 1.5x gate."""

    baseline = _finite_positive(baseline_s, "baseline_s")
    candidate = _finite_positive(candidate_s, "candidate_s")
    orders = [str(order) for order in orders]
    if len(baseline) != len(candidate) or len(baseline) != len(orders):
        raise ValueError("baseline, candidate and order lengths must match")
    if any(order not in {"AB", "BA"} for order in orders):
        raise ValueError("orders must contain only AB or BA")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    ratios = np.asarray(baseline, dtype=np.float64) / np.asarray(
        candidate, dtype=np.float64
    )
    log_ratios = np.log(ratios)
    groups = [
        np.flatnonzero(np.asarray(orders) == order) for order in ("AB", "BA")
    ]
    groups = [group for group in groups if group.size]
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        selected = np.concatenate(
            [rng.choice(group, size=group.size, replace=True) for group in groups]
        )
        bootstrap[sample_index] = math.exp(
            float(np.median(log_ratios[selected]))
        )
    lower_99 = float(np.quantile(bootstrap, 0.01, method="lower"))
    minimum = float(np.min(ratios))
    ratio_of_medians = statistics.median(baseline) / statistics.median(
        candidate
    )
    paired_median = float(np.median(ratios))
    successes = int(np.count_nonzero(ratios > threshold))
    passes = (
        ratio_of_medians >= threshold
        and lower_99 >= threshold
        and minimum >= threshold
    )
    return {
        "threshold": float(threshold),
        "ratio_of_medians": float(ratio_of_medians),
        "paired_median_speedup": paired_median,
        "minimum_paired_speedup": minimum,
        "maximum_paired_speedup": float(np.max(ratios)),
        "paired_speedups": [float(value) for value in ratios],
        "conservative_envelope_speedup": float(
            min(baseline) / max(candidate)
        ),
        "paired_bootstrap_lower_99": lower_99,
        "bootstrap_method": (
            "one-sided 1% percentile of stratified AB/BA paired log-ratio "
            "median bootstrap"
        ),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_samples": int(bootstrap_samples),
        "pairs_strictly_above_threshold": successes,
        "pair_count": len(baseline),
        "one_sided_sign_test_p_at_threshold": float(
            _binomial_upper_tail(successes, len(baseline))
        ),
        "passes_frozen_speed_gate": bool(passes),
    }


def decide_speed_proof(report: dict) -> dict:
    runtime = report.get("context", {}).get("runtime", {})
    cases = report.get("cases", {})
    complete = set(cases) == set(CASE_NAMES)
    expected = [cases.get(name, {}) for name in CASE_NAMES]
    gates = {
        "eight_tpu_devices": runtime.get("active_device_count")
        == TARGET_DEVICE_COUNT,
        "all_devices_are_tpu": runtime.get("all_devices_are_tpu") is True,
        "all_six_cases_present": complete,
        "all_cases_bitwise_exact": complete
        and all(case.get("exact") is True for case in expected),
        "all_cases_pass_frozen_speed_gate": complete
        and all(
            case.get("timing", {}).get("passes_frozen_speed_gate") is True
            for case in expected
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    observed = [
        case.get("timing", {}).get("ratio_of_medians")
        for case in expected
        if isinstance(case.get("timing", {}).get("ratio_of_medians"), (int, float))
    ]
    lower_bounds = [
        case.get("timing", {}).get("paired_bootstrap_lower_99")
        for case in expected
        if isinstance(
            case.get("timing", {}).get("paired_bootstrap_lower_99"),
            (int, float),
        )
    ]
    minimum_pairs = [
        case.get("timing", {}).get("minimum_paired_speedup")
        for case in expected
        if isinstance(
            case.get("timing", {}).get("minimum_paired_speedup"),
            (int, float),
        )
    ]
    return {
        "publishable": not failed,
        "gates": gates,
        "failed_gates": failed,
        "frozen_minimum_speedup": MIN_SPEEDUP,
        "minimum_case_ratio_of_medians": min(observed) if observed else None,
        "minimum_case_bootstrap_lower_99": (
            min(lower_bounds) if lower_bounds else None
        ),
        "minimum_observed_pair_speedup": (
            min(minimum_pairs) if minimum_pairs else None
        ),
    }


def _series_summary(values: list[float], global_batch: int) -> dict:
    return {
        "samples_s": values,
        "median_s": float(statistics.median(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
        "median_global_states_per_s": float(
            global_batch / statistics.median(values)
        ),
    }


def _measure_case(
    original_call,
    exact_call,
    *,
    case_index: int,
    bootstrap_seed: int,
) -> tuple[object, object, dict]:
    calls = {"original_jax": original_call, "exact_split": exact_call}
    first = {}
    for name in ("original_jax", "exact_split"):
        started = time.perf_counter()
        output = jax.block_until_ready(calls[name]())
        first[name] = {
            "s": time.perf_counter() - started,
            "output": output,
        }

    warmup_order = []
    for warmup in range(WARMUPS):
        order = ("original_jax", "exact_split")
        if (warmup + case_index) % 2:
            order = tuple(reversed(order))
        for name in order:
            jax.block_until_ready(calls[name]())
        warmup_order.append("AB" if order[0] == "original_jax" else "BA")

    samples = {"original_jax": [], "exact_split": []}
    pairs = []
    orders = []
    for repeat in range(REPEATS):
        order = ("original_jax", "exact_split")
        if (repeat + case_index) % 2:
            order = tuple(reversed(order))
        order_name = "AB" if order[0] == "original_jax" else "BA"
        pair = {"repeat": repeat, "order": order_name}
        for name in order:
            started = time.perf_counter()
            jax.block_until_ready(calls[name]())
            elapsed = time.perf_counter() - started
            samples[name].append(elapsed)
            pair[f"{name}_s"] = elapsed
        pair["speedup"] = pair["original_jax_s"] / pair["exact_split_s"]
        pairs.append(pair)
        orders.append(order_name)

    stats = paired_speed_statistics(
        baseline_s=samples["original_jax"],
        candidate_s=samples["exact_split"],
        orders=orders,
        threshold=MIN_SPEEDUP,
        bootstrap_seed=bootstrap_seed,
    )
    stats.update(
        original_jax={
            "first_compile_and_execute_s": first["original_jax"]["s"],
            **_series_summary(samples["original_jax"], GLOBAL_BATCH),
        },
        exact_split={
            "first_compile_and_execute_s": first["exact_split"]["s"],
            **_series_summary(samples["exact_split"], GLOBAL_BATCH),
        },
        warmups=WARMUPS,
        repeats=REPEATS,
        warmup_order=warmup_order,
        pairs=pairs,
    )
    return first["original_jax"]["output"], first["exact_split"]["output"], stats


def _make_states(puzzle, kind: str, seed: int) -> np.ndarray:
    if kind == "legal":
        return make_legal_scrambles(
            puzzle, batch=GLOBAL_BATCH, seed=seed
        ).states
    if kind == "stress":
        return np.random.default_rng(seed).integers(
            0, 150, (GLOBAL_BATCH, 150), dtype=np.uint8
        )
    raise ValueError(f"unknown corpus kind: {kind}")


def run_speed_proof(*, dataset: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "claim": "exact full-Q inference is at least 1.5x faster",
            "scope": (
                "warmed synchronized device execution with device-resident "
                "inputs and weights; excludes compile and placement"
            ),
            "target_devices": TARGET_DEVICE_COUNT,
            "local_batch_per_device": LOCAL_BATCH,
            "global_batch": GLOBAL_BATCH,
            "warmups": WARMUPS,
            "paired_repeats": REPEATS,
            "minimum_speedup": MIN_SPEEDUP,
            "case_definitions": [
                {"name": name, "kind": kind, "seed": seed}
                for name, kind, seed in CASE_DEFINITIONS
            ],
        },
        "context": {},
        "cases": {},
        "decision": {},
    }
    checkpoint(result_path, report)
    try:
        devices = jax.devices()
        selected = devices[:TARGET_DEVICE_COUNT]
        inventory = runtime_inventory()
        inventory.update(
            active_device_count=len(selected),
            all_devices_are_tpu=(
                len(selected) == TARGET_DEVICE_COUNT
                and all(device.platform == "tpu" for device in selected)
            ),
        )
        if len(selected) != TARGET_DEVICE_COUNT or not inventory[
            "all_devices_are_tpu"
        ]:
            raise RuntimeError(f"requires eight TPU devices, found: {devices}")
        source_commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip()
        checkpoint_path = dataset / "q555_2k_BEST.pt"
        model_source = dataset / "jax_model.py"
        puzzle_path = dataset / "puzzle_info.json"
        expected_model_hash = (
            "6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e"
        )
        model_hash = sha256_file(model_source)
        if model_hash != expected_model_hash:
            raise RuntimeError(f"Artgor jax_model.py hash changed: {model_hash}")

        exact_config = ArtgorExactConfig(
            prefix_bm=4096,
            head_bm=256,
            head_bk=1024,
            head_bn=128,
            dense_rounding="late",
            inference_chunk=LOCAL_BATCH,
            parent_chunk=131_072,
        )
        report["context"] = {
            "source_commit": source_commit,
            "artgor_script_version": 344319112,
            "runtime": inventory,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_source_sha256": model_hash,
            "puzzle_sha256": sha256_file(puzzle_path),
            "exact_config": dataclasses.asdict(exact_config),
        }
        checkpoint(result_path, report)
        print(json.dumps(report["context"], indent=2), flush=True)

        sys.path.insert(0, str(dataset))
        from jax_model import apply as original_apply, load_params_from_pt

        with jax.default_device(jax.local_devices()[0]):
            params = load_params_from_pt(checkpoint_path)
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        mesh = Mesh(np.asarray(selected), ("core",))
        runtime = prepare_artgor_exact_beam_runtime(
            params,
            mesh=mesh,
            exact_config=exact_config,
            state_storage_len=150,
        )
        original_inference, original_weights_d = _make_original_inference(
            original_apply, params, mesh
        )
        exact_weights_d = _replicate(runtime.weights, mesh)
        state_sharding = NamedSharding(mesh, P("core", None))

        for case_index, (name, kind, seed) in enumerate(CASE_DEFINITIONS):
            states_host = _make_states(puzzle, kind, seed)
            states_d = jax.device_put(states_host, state_sharding)
            original_output, exact_output, timing = _measure_case(
                lambda: original_inference(states_d, original_weights_d),
                lambda: runtime.inference(states_d, exact_weights_d),
                case_index=case_index,
                bootstrap_seed=10_000 + seed,
            )
            comparison = _tensor_comparison(original_output, exact_output)
            report["cases"][name] = {
                "kind": kind,
                "seed": seed,
                "input_sha256": _array_sha256(states_host),
                "global_batch": GLOBAL_BATCH,
                "local_batch_per_device": LOCAL_BATCH,
                **comparison,
                "timing": timing,
            }
            checkpoint(result_path, report)
            print(
                f"CASE {name}: exact={comparison['exact']} "
                f"median={timing['ratio_of_medians']:.4f}x "
                f"lower99={timing['paired_bootstrap_lower_99']:.4f}x "
                f"min_pair={timing['minimum_paired_speedup']:.4f}x",
                flush=True,
            )
            del states_host, states_d, original_output, exact_output

        report["decision"] = decide_speed_proof(report)
        report["status"] = (
            "complete" if report["decision"]["publishable"] else "rejected"
        )
        checkpoint(result_path, report)
        return report
    except Exception as error:
        report.update(
            status="error",
            fatal_error_type=type(error).__name__,
            fatal_error=str(error),
            fatal_traceback=traceback.format_exc(),
        )
        report["decision"] = decide_speed_proof(report)
        checkpoint(result_path, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = run_speed_proof(
        dataset=_dataset_path(args.dataset),
        output=args.output,
    )
    print("DECISION", json.dumps(report["decision"]), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()

