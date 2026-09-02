"""Correctness and promotion gates for full all-Pallas Artgor inference."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
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
from benchmarks.artgor_exact_speed_proof import paired_speed_statistics
from benchmarks.layernorm_quality import load_puzzle, make_legal_scrambles
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.artgor_staged_beam import (
    ArtgorExactConfig,
    prepare_artgor_exact_beam_runtime,
)
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    PallasExactConfig,
    make_sharded_pallas_exact_inference,
    pallas_exact_custom_call_count,
    pallas_exact_stage_names,
    prepare_pallas_exact_weights,
    stream1_layernorm_pallas_exact_stages,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference,
    layernorm_stream1_weights_from_artgor_params,
)


CASE_DEFINITIONS = (
    ("legal_seed_42", "legal", 42),
    ("legal_seed_142", "legal", 142),
    ("legal_seed_242", "legal", 242),
    ("stress_seed_43", "stress", 43),
    ("stress_seed_143", "stress", 143),
    ("stress_seed_243", "stress", 243),
)
CASE_NAMES = tuple(name for name, _, _ in CASE_DEFINITIONS)
TARGET_DEVICE_COUNT = 8
MIN_HYBRID_SPEEDUP = 1.0
STAGE_LOCAL_BATCH = 256
SCREEN_LOCAL_BATCH = 16_384
CONFIRMATION_LOCAL_BATCH = 32_768
WARMUPS = 3
REPEATS = 21
RESULT_NAME = "artgor_pallas_exact_diagnostic.json"

_FORBIDDEN_MODEL_OPS = (
    "stablehlo.dot_general",
    "stablehlo.gather",
    "stablehlo.reduce",
    "stablehlo.maximum",
)


def candidate_configs() -> dict[str, PallasExactConfig]:
    """Correctness-first split-mean candidate; BK1024 remains a negative control."""

    common = dict(
        embedding_bm=4096,
        input_bm=128,
        input_bn=256,
        residual_bm=128,
        residual_bn=256,
        head_bm=256,
        head_bk=1024,
        head_bn=128,
        dense_rounding="late",
        layernorm_arithmetic="split_mean_hlo_mixed",
    )
    return {
        "pallas_exact_split_mean_bk128": PallasExactConfig(
            input_bk=128, residual_bk=128, **common,
        ),
    }


def reference_stage_sequence(states, weights, architecture):
    """Expose the original typed-JAX operator boundaries without changing order."""

    logical = states[:, : architecture.STATE_LEN]
    stages = []
    hidden = weights.embedding[logical.astype(jnp.int32)].reshape(
        states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM
    )
    stages.append(("embedding", hidden))

    def dense(values, layer):
        return values @ layer.weight + layer.bias

    def normalize(values, layer, *, relu):
        result = layer_norm_reference(
            values,
            layer.normalization,
            epsilon=architecture.LAYER_NORM_EPSILON,
        )
        return jax.nn.relu(result) if relu else result

    hidden = dense(hidden, weights.input.dense)
    stages.append(("input.dense", hidden))
    hidden = normalize(hidden, weights.input, relu=True)
    stages.append(("input.layernorm_relu", hidden))
    for index, block in enumerate(weights.residuals):
        skip = hidden
        branch = dense(hidden, block.first.dense)
        stages.append((f"residual.{index}.dense1", branch))
        branch = normalize(branch, block.first, relu=True)
        stages.append((f"residual.{index}.layernorm1_relu", branch))
        branch = dense(branch, block.second.dense)
        stages.append((f"residual.{index}.dense2", branch))
        hidden = jax.nn.relu(skip + normalize(branch, block.second, relu=False))
        stages.append((f"residual.{index}.layernorm2_skip_relu", hidden))
    output = dense(hidden, weights.output)
    stages.append(("head.dense", output))
    expected_names = pallas_exact_stage_names(architecture)
    if tuple(name for name, _ in stages) != expected_names:
        raise AssertionError("reference stage builder drifted from the model contract")
    return tuple(stages)


def _sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def compare_stage_sequences(reference, candidate) -> dict:
    """Compare named operator boundaries directly and stop attribution at first drift."""

    reference = list(reference)
    candidate = list(candidate)
    reference_names = tuple(name for name, _ in reference)
    candidate_names = tuple(name for name, _ in candidate)
    if reference_names != candidate_names:
        raise ValueError("reference and candidate stage names must match exactly")
    rows = []
    first_mismatch = None
    for (name, reference_value), (_, candidate_value) in zip(reference, candidate):
        expected = np.asarray(reference_value)
        actual = np.asarray(candidate_value)
        if expected.shape != actual.shape:
            raise ValueError(f"stage {name} shape mismatch: {expected.shape} != {actual.shape}")
        exact_mask = expected == actual
        exact = bool(np.all(exact_mask))
        mismatch_count = int(exact_mask.size - np.count_nonzero(exact_mask))
        if not exact and first_mismatch is None:
            first_mismatch = name
        difference = np.abs(
            expected.astype(np.float32) - actual.astype(np.float32)
        )
        rows.append({
            "name": name,
            "shape": list(expected.shape),
            "finite": bool(
                np.isfinite(expected.astype(np.float32)).all()
                and np.isfinite(actual.astype(np.float32)).all()
            ),
            "exact": exact,
            "mismatch_count": mismatch_count,
            "max_abs": float(np.max(difference, initial=0.0)),
            "mean_abs": float(np.mean(difference)) if difference.size else 0.0,
            "reference_sha256": _sha256(expected),
            "candidate_sha256": _sha256(actual),
        })
    return {
        "stage_count": len(rows),
        "all_stages_exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "stages": rows,
    }


def audit_all_pallas_hlo(
    hlo_text: str,
    stage_names,
    *,
    expected_custom_call_count: int | None = None,
) -> dict:
    """Reject model arithmetic that escaped the declared Pallas custom calls."""

    text = str(hlo_text)
    semantic_count = len(tuple(stage_names))
    expected_count = (
        semantic_count
        if expected_custom_call_count is None
        else expected_custom_call_count
    )
    custom_call_count = text.count("tpu_custom_call")
    forbidden = [operation for operation in _FORBIDDEN_MODEL_OPS if operation in text]
    return {
        "expected_stage_count": semantic_count,
        "expected_custom_call_count": expected_count,
        "custom_call_count": custom_call_count,
        "forbidden_operations": forbidden,
        "passes": custom_call_count == expected_count and not forbidden,
    }


def decide_pallas_exact(report: dict) -> dict:
    """Apply the frozen exactness, implementation and speed promotion gates."""

    runtime = report.get("context", {}).get("runtime", {})
    cases = report.get("cases", {})
    complete = set(cases) == set(CASE_NAMES)
    ordered = [cases.get(name, {}) for name in CASE_NAMES]
    gates = {
        "eight_tpu_devices": (
            runtime.get("active_device_count") == TARGET_DEVICE_COUNT
        ),
        "all_devices_are_tpu": runtime.get("all_devices_are_tpu") is True,
        "all_six_cases_present": complete,
        "all_operator_boundaries_bitwise_exact": complete and all(
            case.get("all_stages_exact") is True for case in ordered
        ),
        "all_full_outputs_bitwise_exact": complete and all(
            case.get("full_output_exact") is True for case in ordered
        ),
        "compiled_hlo_is_all_pallas": (
            report.get("hlo_audit", {}).get("passes") is True
        ),
        "all_cases_beat_exact_hybrid": complete and all(
            case.get("timing", {}).get("passes_frozen_speed_gate") is True
            for case in ordered
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "promote": not failed,
        "gates": gates,
        "failed_gates": failed,
        "speed_baseline": "exact_split",
        "minimum_hybrid_speedup": MIN_HYBRID_SPEEDUP,
    }


def _make_states(puzzle, kind: str, seed: int, global_batch: int) -> np.ndarray:
    if kind == "legal":
        return make_legal_scrambles(
            puzzle, batch=global_batch, seed=seed,
        ).states
    if kind == "stress":
        return np.random.default_rng(seed).integers(
            0, 150, (global_batch, 150), dtype=np.uint8,
        )
    raise ValueError(f"unknown corpus kind: {kind}")


def _mapped_stages(local_call, *, mesh, weights_example, stage_count):
    weight_specs = jax.tree.map(lambda _: P(), weights_example)
    return jax.jit(jax.shard_map(
        local_call,
        mesh=mesh,
        in_specs=(P("core", None), weight_specs),
        out_specs=tuple(P("core", None) for _ in range(stage_count)),
        check_vma=False,
    ))


def _series(values, global_batch):
    return {
        "samples_s": [float(value) for value in values],
        "median_s": float(statistics.median(values)),
        "median_global_states_per_s": float(
            global_batch / statistics.median(values)
        ),
    }


def _measure_three(calls, *, case_index, global_batch, bootstrap_seed):
    names = ("original_jax", "exact_split", "pallas_exact")
    first = {}
    for name in names:
        started = time.perf_counter()
        value = jax.block_until_ready(calls[name]())
        first[name] = {
            "s": time.perf_counter() - started,
            "value": value,
        }
    for warmup in range(WARMUPS):
        order = names if (warmup + case_index) % 2 == 0 else tuple(reversed(names))
        for name in order:
            jax.block_until_ready(calls[name]())
    samples = {name: [] for name in names}
    pairs = []
    pair_orders = []
    for repeat in range(REPEATS):
        order = names if (repeat + case_index) % 2 == 0 else tuple(reversed(names))
        row = {"repeat": repeat, "order": list(order)}
        for name in order:
            started = time.perf_counter()
            jax.block_until_ready(calls[name]())
            elapsed = time.perf_counter() - started
            samples[name].append(elapsed)
            row[f"{name}_s"] = elapsed
        pair_order = (
            "AB" if order.index("exact_split") < order.index("pallas_exact")
            else "BA"
        )
        pair_orders.append(pair_order)
        row["hybrid_over_pallas"] = (
            row["exact_split_s"] / row["pallas_exact_s"]
        )
        pairs.append(row)
    vs_hybrid = paired_speed_statistics(
        baseline_s=samples["exact_split"],
        candidate_s=samples["pallas_exact"],
        orders=pair_orders,
        threshold=MIN_HYBRID_SPEEDUP,
        bootstrap_seed=bootstrap_seed,
    )
    vs_original = paired_speed_statistics(
        baseline_s=samples["original_jax"],
        candidate_s=samples["pallas_exact"],
        orders=pair_orders,
        threshold=1.5,
        bootstrap_seed=bootstrap_seed + 1,
    )
    return {
        "first_execute_s": {name: first[name]["s"] for name in names},
        "series": {name: _series(samples[name], global_batch) for name in names},
        "vs_exact_split": vs_hybrid,
        "vs_original_jax": vs_original,
        "pairs": pairs,
        "warmups": WARMUPS,
        "repeats": REPEATS,
    }, {name: first[name]["value"] for name in names}


def _stage_diagnostics(
    *, puzzle, mesh, state_sharding, reference_runner, reference_weights_d,
    candidate_runners, candidate_weights_d,
):
    stage_names = None
    results = {identifier: {} for identifier in candidate_runners}
    global_batch = TARGET_DEVICE_COUNT * STAGE_LOCAL_BATCH
    for name, kind, seed in CASE_DEFINITIONS:
        states_host = _make_states(puzzle, kind, seed, global_batch)
        states_d = jax.device_put(states_host, state_sharding)
        reference_values = jax.block_until_ready(
            reference_runner(states_d, reference_weights_d)
        )
        for identifier, runner in candidate_runners.items():
            candidate_values = jax.block_until_ready(
                runner(states_d, candidate_weights_d)
            )
            if stage_names is None:
                stage_names = tuple(
                    f"stage_{index}" for index in range(len(reference_values))
                )
            named_reference = zip(
                pallas_exact_stage_names_from_count(len(reference_values)),
                [np.asarray(value) for value in reference_values],
            )
            named_candidate = zip(
                pallas_exact_stage_names_from_count(len(candidate_values)),
                [np.asarray(value) for value in candidate_values],
            )
            comparison = compare_stage_sequences(named_reference, named_candidate)
            comparison.update(
                kind=kind,
                seed=seed,
                input_sha256=_array_sha256(states_host),
                global_batch=global_batch,
                local_batch_per_device=STAGE_LOCAL_BATCH,
            )
            results[identifier][name] = comparison
        del states_host, states_d, reference_values
    return results


def pallas_exact_stage_names_from_count(stage_count: int) -> tuple[str, ...]:
    residual_count, remainder = divmod(stage_count - 4, 4)
    if stage_count < 4 or remainder:
        raise ValueError("stage count must satisfy 4*N+4")
    class Contract:
        RESIDUAL_COUNT = residual_count
    return pallas_exact_stage_names(Contract())


def _full_cases(
    *, puzzle, state_sharding, original_call, original_weights_d,
    hybrid_call, hybrid_weights_d, candidate_call, candidate_weights_d,
    local_batch, phase, candidate_id,
):
    rows = {}
    global_batch = TARGET_DEVICE_COUNT * local_batch
    for case_index, (name, kind, seed) in enumerate(CASE_DEFINITIONS):
        states_host = _make_states(puzzle, kind, seed, global_batch)
        states_d = jax.device_put(states_host, state_sharding)
        timing, outputs = _measure_three(
            {
                "original_jax": lambda: original_call(states_d, original_weights_d),
                "exact_split": lambda: hybrid_call(states_d, hybrid_weights_d),
                "pallas_exact": lambda: candidate_call(states_d, candidate_weights_d),
            },
            case_index=case_index,
            global_batch=global_batch,
            bootstrap_seed=20_000 + seed + local_batch,
        )
        original_vs_candidate = _tensor_comparison(
            outputs["original_jax"], outputs["pallas_exact"]
        )
        hybrid_vs_candidate = _tensor_comparison(
            outputs["exact_split"], outputs["pallas_exact"]
        )
        rows[name] = {
            "phase": phase,
            "candidate_id": candidate_id,
            "kind": kind,
            "seed": seed,
            "input_sha256": _array_sha256(states_host),
            "global_batch": global_batch,
            "local_batch_per_device": local_batch,
            "full_output_exact": original_vs_candidate["exact"],
            "original_vs_pallas": original_vs_candidate,
            "hybrid_vs_pallas": hybrid_vs_candidate,
            "timing": timing["vs_exact_split"],
            "timing_vs_original": timing["vs_original_jax"],
            "timing_detail": timing,
        }
        del states_host, states_d, outputs
    return rows


def run_diagnostic(*, dataset: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "scope": "full state-to-30Q inference only; beam stages excluded",
            "stage_local_batch_per_device": STAGE_LOCAL_BATCH,
            "screen_local_batch_per_device": SCREEN_LOCAL_BATCH,
            "confirmation_local_batch_per_device": CONFIRMATION_LOCAL_BATCH,
            "warmups": WARMUPS,
            "paired_repeats": REPEATS,
            "promotion": (
                "all 44 boundaries and full output bitwise exact on six corpora; "
                "clean outer HLO; every paired hybrid/pallas ratio and lower99 >=1"
            ),
        },
        "context": {},
        "candidates": {},
        "screen_cases": {},
        "cases": {},
        "hlo_audit": {},
        "decision": {},
    }
    checkpoint(result_path, report)
    try:
        devices = jax.devices()
        selected_devices = devices[:TARGET_DEVICE_COUNT]
        inventory = runtime_inventory()
        inventory.update(
            active_device_count=len(selected_devices),
            all_devices_are_tpu=(
                len(selected_devices) == TARGET_DEVICE_COUNT
                and all(device.platform == "tpu" for device in selected_devices)
            ),
        )
        if not inventory["all_devices_are_tpu"]:
            raise RuntimeError(f"requires eight TPU devices, found: {devices}")
        source_commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True,
        ).strip()
        checkpoint_path = dataset / "q555_2k_BEST.pt"
        model_path = dataset / "jax_model.py"
        puzzle_path = dataset / "puzzle_info.json"
        report["context"] = {
            "source_commit": source_commit,
            "runtime": inventory,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_source_sha256": sha256_file(model_path),
            "puzzle_sha256": sha256_file(puzzle_path),
            "candidate_configs": {
                name: dataclasses.asdict(config)
                for name, config in candidate_configs().items()
            },
        }
        checkpoint(result_path, report)
        sys.path.insert(0, str(dataset))
        from jax_model import apply as original_apply, load_params_from_pt

        with jax.default_device(jax.local_devices()[0]):
            params = load_params_from_pt(checkpoint_path)
        architecture = Stream1Architecture.from_artgor_params(
            params, STATE_STORAGE_LEN=150,
        )
        if (
            architecture.STATE_LEN,
            architecture.NUM_CLASSES,
            architecture.EMBED_DIM,
            architecture.HIDDEN1,
            architecture.RESIDUAL_COUNT,
            architecture.MOVE_COUNT,
        ) != (150, 150, 24, 1024, 10, 30):
            raise RuntimeError("checkpoint is not the frozen Artgor Q ResMLP")
        typed_weights = layernorm_stream1_weights_from_artgor_params(
            params, architecture,
        )
        candidate_weights = prepare_pallas_exact_weights(
            typed_weights, architecture,
        )
        exact_config = ArtgorExactConfig()
        mesh = Mesh(np.asarray(selected_devices), ("core",))
        state_sharding = NamedSharding(mesh, P("core", None))
        original_runner, original_weights_d = _make_original_inference(
            original_apply, params, mesh,
        )
        hybrid_runtime = prepare_artgor_exact_beam_runtime(
            params,
            mesh=mesh,
            exact_config=exact_config,
            state_storage_len=150,
        )
        hybrid_weights_d = _replicate(hybrid_runtime.weights, mesh)
        typed_weights_d = _replicate(typed_weights, mesh)
        candidate_weights_d = _replicate(candidate_weights, mesh)
        names = pallas_exact_stage_names(architecture)
        reference_stage_runner = _mapped_stages(
            lambda states, weights: tuple(
                value for _, value in reference_stage_sequence(
                    states, weights, architecture,
                )
            ),
            mesh=mesh,
            weights_example=typed_weights,
            stage_count=len(names),
        )
        configs = candidate_configs()
        candidate_stage_runners = {
            identifier: _mapped_stages(
                lambda states, weights, config=config: tuple(
                    stage.value for stage in stream1_layernorm_pallas_exact_stages(
                        states,
                        weights,
                        architecture,
                        config=config,
                    )
                ),
                mesh=mesh,
                weights_example=candidate_weights,
                stage_count=len(names),
            )
            for identifier, config in configs.items()
        }
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        stage_results = _stage_diagnostics(
            puzzle=puzzle,
            mesh=mesh,
            state_sharding=state_sharding,
            reference_runner=reference_stage_runner,
            reference_weights_d=typed_weights_d,
            candidate_runners=candidate_stage_runners,
            candidate_weights_d=candidate_weights_d,
        )
        for identifier, config in configs.items():
            report["candidates"][identifier] = {
                "config": dataclasses.asdict(config),
                "stage_cases": stage_results[identifier],
                "all_stage_cases_exact": all(
                    row["all_stages_exact"] for row in stage_results[identifier].values()
                ),
            }
        checkpoint(result_path, report)

        eligible = [
            identifier for identifier, row in report["candidates"].items()
            if row["all_stage_cases_exact"]
        ]
        if not eligible:
            report["status"] = "rejected"
            report["decision"] = {
                "promote": False,
                "failed_gates": ["no_stage_exact_candidate"],
                "first_mismatches": {
                    identifier: {
                        case: row["first_mismatch"]
                        for case, row in candidate["stage_cases"].items()
                    }
                    for identifier, candidate in report["candidates"].items()
                },
            }
            checkpoint(result_path, report)
            return report

        candidate_full_runners = {
            identifier: make_sharded_pallas_exact_inference(
                architecture,
                mesh=mesh,
                weights_example=candidate_weights,
                config=configs[identifier],
            )
            for identifier in eligible
        }
        for identifier, runner in candidate_full_runners.items():
            rows = _full_cases(
                puzzle=puzzle,
                state_sharding=state_sharding,
                original_call=original_runner,
                original_weights_d=original_weights_d,
                hybrid_call=hybrid_runtime.inference,
                hybrid_weights_d=hybrid_weights_d,
                candidate_call=runner,
                candidate_weights_d=candidate_weights_d,
                local_batch=SCREEN_LOCAL_BATCH,
                phase="screen",
                candidate_id=identifier,
            )
            report["screen_cases"][identifier] = rows
            checkpoint(result_path, report)
        full_exact = [
            identifier for identifier in eligible
            if all(
                row["full_output_exact"]
                for row in report["screen_cases"][identifier].values()
            )
        ]
        if not full_exact:
            report["status"] = "rejected"
            report["decision"] = {
                "promote": False,
                "failed_gates": ["no_full_output_exact_candidate"],
            }
            checkpoint(result_path, report)
            return report
        selected_id = max(
            full_exact,
            key=lambda identifier: min(
                row["timing"]["ratio_of_medians"]
                for row in report["screen_cases"][identifier].values()
            ),
        )
        selected_runner = candidate_full_runners[selected_id]
        report["selected_id"] = selected_id
        report["cases"] = _full_cases(
            puzzle=puzzle,
            state_sharding=state_sharding,
            original_call=original_runner,
            original_weights_d=original_weights_d,
            hybrid_call=hybrid_runtime.inference,
            hybrid_weights_d=hybrid_weights_d,
            candidate_call=selected_runner,
            candidate_weights_d=candidate_weights_d,
            local_batch=CONFIRMATION_LOCAL_BATCH,
            phase="confirmation",
            candidate_id=selected_id,
        )
        for name in CASE_NAMES:
            report["cases"][name]["all_stages_exact"] = report[
                "candidates"
            ][selected_id]["stage_cases"][name]["all_stages_exact"]
        sample_host = _make_states(
            puzzle, "legal", 42, TARGET_DEVICE_COUNT * SCREEN_LOCAL_BATCH,
        )
        sample_d = jax.device_put(sample_host, state_sharding)
        lowered = selected_runner.lower(sample_d, candidate_weights_d)
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        hlo_path = output / "pallas_exact_full.stablehlo.txt"
        hlo_path.write_text(stablehlo, encoding="utf-8")
        report["hlo_audit"] = audit_all_pallas_hlo(
            stablehlo,
            names,
            expected_custom_call_count=pallas_exact_custom_call_count(
                architecture, configs[selected_id],
            ),
        )
        report["hlo_audit"]["stablehlo_sha256"] = hashlib.sha256(
            stablehlo.encode()
        ).hexdigest()
        report["decision"] = decide_pallas_exact(report)
        report["status"] = (
            "complete" if report["decision"]["promote"] else "rejected"
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
        checkpoint(result_path, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_diagnostic(
        dataset=_dataset_path(args.dataset), output=args.output,
    )
    print("DECISION", json.dumps(report["decision"], allow_nan=False), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
