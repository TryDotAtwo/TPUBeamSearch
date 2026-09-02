"""Subtraction-only TPU attribution after exact LayerNorm mean."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.artgor_exact_notebook_validation import _dataset_path, _replicate, checkpoint
from benchmarks.artgor_pallas_exact_diagnostic import _make_states
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import flat_embedding_prepacked
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_pallas_exact import prepare_pallas_exact_weights
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)
from tpu_beam_search.stream1_layernorm_subtraction import (
    pallas_centered_subtraction,
    pallas_centered_variance,
)


CASE_DEFINITIONS = (
    ("legal_seed_42", "legal", 42),
    ("legal_seed_142", "legal", 142),
    ("legal_seed_242", "legal", 242),
    ("stress_seed_43", "stress", 43),
    ("stress_seed_143", "stress", 143),
    ("stress_seed_243", "stress", 243),
)
RESULT_NAME = "artgor_layernorm_subtraction.json"
TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 256


def _sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def tensor_metrics(reference, candidate, *, witness_limit: int = 8) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("tensor shapes must match")
    reference_fp64 = reference.astype(np.float64)
    candidate_fp64 = candidate.astype(np.float64)
    difference = candidate_fp64 - reference_fp64
    mismatch_indices = np.flatnonzero(reference.reshape(-1) != candidate.reshape(-1))
    witnesses = [
        {
            "flat_index": int(index),
            "reference": float(reference.reshape(-1)[index]),
            "candidate": float(candidate.reshape(-1)[index]),
        }
        for index in mismatch_indices[:witness_limit]
    ]
    return {
        "shape": list(reference.shape),
        "finite": bool(np.isfinite(reference_fp64).all() and np.isfinite(candidate_fp64).all()),
        "mismatch_count": int(mismatch_indices.size),
        "exact": mismatch_indices.size == 0,
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "reference_sha256": _sha256(reference),
        "candidate_sha256": _sha256(candidate),
        "witnesses": witnesses,
    }


def _mapped(local_call, *, mesh, weight_example):
    from jax.sharding import PartitionSpec as P

    weight_specs = jax.tree.map(lambda _: P(), weight_example)
    return jax.jit(jax.shard_map(
        local_call,
        mesh=mesh,
        in_specs=(P("core", None), weight_specs),
        out_specs=P("core", None),
        check_vma=False,
    ))


def _mapped_binary(local_call, *, mesh):
    from jax.sharding import PartitionSpec as P

    return jax.jit(jax.shard_map(
        local_call,
        mesh=mesh,
        in_specs=(P("core", None), P("core", None)),
        out_specs=P("core", None),
        check_vma=False,
    ))


def _hlo_identity(call, *args) -> dict[str, Any]:
    text = str(call.lower(*args).compiler_ir(dialect="stablehlo"))
    return {
        "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "stablehlo_bytes": len(text.encode()),
        "tpu_custom_call_count": text.count("tpu_custom_call"),
    }


def run_subtraction(*, dataset: Path, output: Path) -> dict:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "scope": "fixed-operand centered subtraction only; beam excluded",
            "local_batch_per_device": LOCAL_BATCH,
            "arms": [
                "jax_same_call",
                "jax_materialized_casts",
                "pallas_real",
                "pallas_interpret",
                "pallas_fused_variance",
            ],
            "oracle": "unchanged monolithic Artgor model remains external oracle",
        },
        "context": {},
        "hlo_identity": {},
        "cases": {},
        "decision": {},
    }
    checkpoint(result_path, report)
    try:
        devices = jax.devices()[:TARGET_DEVICE_COUNT]
        inventory = runtime_inventory()
        inventory.update(
            active_device_count=len(devices),
            all_devices_are_tpu=(
                len(devices) == TARGET_DEVICE_COUNT
                and all(device.platform == "tpu" for device in devices)
            ),
        )
        if not inventory["all_devices_are_tpu"]:
            raise RuntimeError(f"requires eight TPU devices, found {jax.devices()}")
        checkpoint_path = dataset / "q555_2k_BEST.pt"
        puzzle_path = dataset / "puzzle_info.json"
        model_path = dataset / "jax_model.py"
        source_commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True,
        ).strip()
        report["context"] = {
            "source_commit": source_commit,
            "runtime": inventory,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_source_sha256": sha256_file(model_path),
            "puzzle_sha256": sha256_file(puzzle_path),
        }
        checkpoint(result_path, report)
        sys.path.insert(0, str(dataset))
        from jax_model import load_params_from_pt

        with jax.default_device(jax.local_devices()[0]):
            params = load_params_from_pt(checkpoint_path)
        architecture = Stream1Architecture.from_artgor_params(
            params, STATE_STORAGE_LEN=150,
        )
        typed = layernorm_stream1_weights_from_artgor_params(params, architecture)
        prepared = prepare_pallas_exact_weights(typed, architecture)
        mesh = Mesh(np.asarray(devices), ("core",))
        state_sharding = NamedSharding(mesh, P("core", None))
        prepared_d = _replicate(prepared, mesh)

        def local_dense(states, weights):
            hidden = flat_embedding_prepacked(
                states[:, : architecture.STATE_LEN], weights.embedding,
                embed_dim=architecture.EMBED_DIM, bm=4096,
            )
            return pallas_layernorm_dense(
                hidden, weights.input.dense.weight, weights.input.dense.bias,
                bm=128, bk=128, bn=256, dense_rounding="late",
            )

        dense_call = _mapped(local_dense, mesh=mesh, weight_example=prepared)
        mean_call = jax.jit(jax.shard_map(
            lambda values: (
                jnp.sum(values.astype(jnp.float32), axis=1, keepdims=True)
                / values.shape[1]
            ).astype(jnp.bfloat16),
            mesh=mesh,
            in_specs=P("core", None),
            out_specs=P("core", None),
            check_vma=False,
        ))
        same_call = _mapped_binary(
            lambda values, mean: (
                values.astype(jnp.float32) - mean.astype(jnp.float32)
            ),
            mesh=mesh,
        )
        cast_call = jax.jit(jax.shard_map(
            lambda value: value.astype(jnp.float32),
            mesh=mesh,
            in_specs=P("core", None),
            out_specs=P("core", None),
            check_vma=False,
        ))
        subtract_casts = _mapped_binary(
            lambda values_fp32, mean_fp32: values_fp32 - mean_fp32,
            mesh=mesh,
        )
        variance_from_centered = jax.jit(jax.shard_map(
            lambda centered: jnp.broadcast_to((
                jnp.sum(centered * centered, axis=1, keepdims=True)
                / centered.shape[1]
            ).astype(jnp.bfloat16), centered.shape),
            mesh=mesh,
            in_specs=P("core", None),
            out_specs=P("core", None),
            check_vma=False,
        ))
        pallas_real = _mapped_binary(
            lambda values, mean: pallas_centered_subtraction(
                values, mean, bm=128,
            ),
            mesh=mesh,
        )
        pallas_interpret = _mapped_binary(
            lambda values, mean: pallas_centered_subtraction(
                values, mean, bm=128, interpret=True,
            ),
            mesh=mesh,
        )
        pallas_variance = _mapped_binary(
            lambda values, mean: pallas_centered_variance(
                values, mean, bm=128,
            ),
            mesh=mesh,
        )
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        global_batch = TARGET_DEVICE_COUNT * LOCAL_BATCH
        for case_index, (case_name, kind, seed) in enumerate(CASE_DEFINITIONS):
            states_host = _make_states(puzzle, kind, seed, global_batch)
            states_d = jax.device_put(states_host, state_sharding)
            dense = jax.block_until_ready(dense_call(states_d, prepared_d))
            mean = jax.block_until_ready(mean_call(dense))
            jax_same = jax.block_until_ready(same_call(dense, mean))
            values_fp32 = jax.block_until_ready(cast_call(dense))
            mean_fp32 = jax.block_until_ready(cast_call(mean))
            jax_materialized = jax.block_until_ready(
                subtract_casts(values_fp32, mean_fp32)
            )
            real = jax.block_until_ready(pallas_real(dense, mean))
            interpreted = jax.block_until_ready(pallas_interpret(dense, mean))
            jax_variance = jax.block_until_ready(
                variance_from_centered(jax_materialized)
            )
            fused_variance = jax.block_until_ready(pallas_variance(dense, mean))
            if case_index == 0:
                report["hlo_identity"] = {
                    "jax_same_call": _hlo_identity(same_call, dense, mean),
                    "jax_cast": _hlo_identity(cast_call, dense),
                    "jax_subtract_casts": _hlo_identity(
                        subtract_casts, values_fp32, mean_fp32,
                    ),
                    "pallas_real": _hlo_identity(pallas_real, dense, mean),
                    "pallas_interpret": _hlo_identity(
                        pallas_interpret, dense, mean,
                    ),
                    "pallas_fused_variance": _hlo_identity(
                        pallas_variance, dense, mean,
                    ),
                }
            report["cases"][case_name] = {
                "kind": kind,
                "seed": seed,
                "global_batch": global_batch,
                "input_sha256": hashlib.sha256(states_host.tobytes()).hexdigest(),
                "dense_sha256": hashlib.sha256(np.asarray(dense).tobytes()).hexdigest(),
                "mean_sha256": hashlib.sha256(np.asarray(mean).tobytes()).hexdigest(),
                "logical_shapes": {
                    "dense": list(dense.shape), "mean": list(mean.shape),
                },
                "centered": {
                    "same_vs_materialized": tensor_metrics(
                        jax_same, jax_materialized,
                    ),
                    "materialized_vs_pallas_real": tensor_metrics(
                        jax_materialized, real,
                    ),
                    "materialized_vs_pallas_interpret": tensor_metrics(
                        jax_materialized, interpreted,
                    ),
                    "interpret_vs_pallas_real": tensor_metrics(interpreted, real),
                },
                "variance": {
                    "materialized_jax_vs_pallas_fused": tensor_metrics(
                        jax_variance, fused_variance,
                    ),
                },
            }
            checkpoint(result_path, report)
        exact_real = all(
            case["centered"]["materialized_vs_pallas_real"]["exact"]
            for case in report["cases"].values()
        )
        exact_interpret = all(
            case["centered"]["materialized_vs_pallas_interpret"]["exact"]
            for case in report["cases"].values()
        )
        report["decision"] = {
            "promote_centered_contract": exact_real,
            "pallas_interpret_matches_materialized_jax": exact_interpret,
            "next": (
                "integrate_centered_contract" if exact_real
                else "attribute_real_vs_interpret_or_jax_boundary"
            ),
        }
        report["status"] = "complete" if exact_real else "rejected"
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_subtraction(
        dataset=_dataset_path(args.dataset), output=args.output,
    )
    print("DECISION", json.dumps(report["decision"], allow_nan=False), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
