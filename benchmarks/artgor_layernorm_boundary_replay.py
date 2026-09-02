"""Replay the five materialized Pallas LayerNorm boundaries on real TPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.artgor_exact_notebook_validation import _dataset_path, _replicate, checkpoint
from benchmarks.artgor_layernorm_attribution import _materialized_jax_controls
from benchmarks.artgor_layernorm_subtraction import CASE_DEFINITIONS, tensor_metrics
from benchmarks.artgor_pallas_exact_diagnostic import _make_states
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import flat_embedding_prepacked
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_pallas_exact import (
    pallas_fully_materialized_layernorm_checkpoints,
    prepare_pallas_exact_weights,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference,
    layernorm_stream1_weights_from_artgor_params,
)


RESULT_NAME = "artgor_layernorm_boundary_replay.json"
TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 256


def replay_checkpoint_names(*, relu: bool) -> tuple[str, ...]:
    return ("mean", "centered", "variance", "invstd", "affine_relu" if relu else "affine_bf16")


def _sequence_metrics(reference: Mapping[str, object], candidate: Mapping[str, object]) -> dict:
    if tuple(reference) != tuple(candidate):
        raise ValueError("checkpoint order differs")
    checkpoints = {
        name: tensor_metrics(reference[name], candidate[name]) for name in reference
    }
    return {
        "first_mismatch": next(
            (name for name, value in checkpoints.items() if value["mismatch_count"]),
            None,
        ),
        "checkpoints": checkpoints,
    }


def compare_replays(*, pallas: Mapping[str, object], materialized: Mapping[str, object], monolithic) -> dict:
    final_name = next(reversed(pallas))
    return {
        "pallas_vs_materialized": _sequence_metrics(materialized, pallas),
        "pallas_vs_monolithic_final": tensor_metrics(monolithic, pallas[final_name]),
        "materialized_vs_monolithic_final": tensor_metrics(monolithic, materialized[final_name]),
    }


def _mapped(local_call, *, mesh, weight_example, output_count=1):
    from jax.sharding import PartitionSpec as P

    output_specs = P("core", None) if output_count == 1 else tuple(
        P("core", None) for _ in range(output_count)
    )
    return jax.jit(jax.shard_map(
        local_call,
        mesh=mesh,
        in_specs=(P("core", None), jax.tree.map(lambda _: P(), weight_example)),
        out_specs=output_specs,
        check_vma=False,
    ))


def run_boundary_replay(*, dataset: Path, output: Path) -> dict:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    names = replay_checkpoint_names(relu=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "scope": "exact BK128 input Dense and first LayerNorm only; beam excluded",
            "local_batch_per_device": LOCAL_BATCH,
            "checkpoint_order": list(names),
            "controls": ["monolithic_jax", "materialized_jax", "modular_real_pallas"],
        },
        "context": {}, "hlo_audit": {}, "cases": {}, "decision": {},
    }
    checkpoint(result_path, report)
    try:
        devices = jax.devices()[:TARGET_DEVICE_COUNT]
        inventory = runtime_inventory()
        inventory.update(
            active_device_count=len(devices),
            all_devices_are_tpu=(len(devices) == TARGET_DEVICE_COUNT and all(d.platform == "tpu" for d in devices)),
        )
        if not inventory["all_devices_are_tpu"]:
            raise RuntimeError(f"requires eight TPU devices, found {jax.devices()}")
        checkpoint_path = dataset / "q555_2k_BEST.pt"
        puzzle_path = dataset / "puzzle_info.json"
        model_path = dataset / "jax_model.py"
        report["context"] = {
            "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
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
        architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=150)
        typed = layernorm_stream1_weights_from_artgor_params(params, architecture)
        prepared = prepare_pallas_exact_weights(typed, architecture)
        mesh = Mesh(np.asarray(devices), ("core",))
        state_sharding = NamedSharding(mesh, P("core", None))
        prepared_d = _replicate(prepared, mesh)
        normalization_d = _replicate(typed.input.normalization, mesh)
        scale_d = _replicate(typed.input.normalization.scale, mesh)
        bias_d = _replicate(typed.input.normalization.bias, mesh)

        def local_dense(states, weights):
            hidden = flat_embedding_prepacked(
                states[:, :architecture.STATE_LEN], weights.embedding,
                embed_dim=architecture.EMBED_DIM, bm=4096,
            )
            return pallas_layernorm_dense(
                hidden, weights.input.dense.weight, weights.input.dense.bias,
                bm=128, bk=128, bn=256, dense_rounding="late",
            )

        dense_call = _mapped(local_dense, mesh=mesh, weight_example=prepared)
        monolithic_call = _mapped(
            lambda values, weights: jax.nn.relu(layer_norm_reference(
                values, weights, epsilon=architecture.LAYER_NORM_EPSILON,
            )),
            mesh=mesh,
            weight_example=typed.input.normalization,
        )

        def local_pallas(values, weights):
            values_by_name = pallas_fully_materialized_layernorm_checkpoints(
                values, weights[0], weights[1], relu=True,
                epsilon=architecture.LAYER_NORM_EPSILON, bm=128,
            )
            return tuple(values_by_name[name] for name in names)

        pallas_call = _mapped(
            local_pallas, mesh=mesh,
            weight_example=(typed.input.normalization.scale, typed.input.normalization.bias),
            output_count=len(names),
        )
        sample_dense = jax.device_put(
            np.zeros((TARGET_DEVICE_COUNT * LOCAL_BATCH, architecture.HIDDEN1), dtype=np.float32).astype(jnp.bfloat16),
            state_sharding,
        )
        lowered = pallas_call.lower(sample_dense, (scale_d, bias_d))
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        report["hlo_audit"] = {
            "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
            "tpu_custom_call_count": stablehlo.count("tpu_custom_call"),
            "expected_tpu_custom_call_count": 5,
        }
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        global_batch = TARGET_DEVICE_COUNT * LOCAL_BATCH
        for case_name, kind, seed in CASE_DEFINITIONS:
            states_host = _make_states(puzzle, kind, seed, global_batch)
            states_d = jax.device_put(states_host, state_sharding)
            dense = jax.block_until_ready(dense_call(states_d, prepared_d))
            monolithic = jax.block_until_ready(monolithic_call(dense, normalization_d))
            materialized_raw = _materialized_jax_controls(
                dense, scale_d, bias_d, mesh=mesh,
                epsilon=architecture.LAYER_NORM_EPSILON,
            )
            materialized = {
                "mean": materialized_raw["mean"],
                "centered": materialized_raw["centered"],
                "variance": materialized_raw["variance"],
                "invstd": materialized_raw["invstd"],
                "affine_relu": materialized_raw["relu"],
            }
            pallas_tuple = jax.block_until_ready(pallas_call(dense, (scale_d, bias_d)))
            pallas = dict(zip(names, pallas_tuple, strict=True))
            report["cases"][case_name] = {
                "kind": kind, "seed": seed, "global_batch": global_batch,
                "input_sha256": hashlib.sha256(states_host.tobytes()).hexdigest(),
                "dense_sha256": hashlib.sha256(np.asarray(dense).tobytes()).hexdigest(),
                "comparisons": compare_replays(
                    pallas=pallas, materialized=materialized, monolithic=monolithic,
                ),
            }
            checkpoint(result_path, report)
        exact_materialized = all(
            case["comparisons"]["pallas_vs_materialized"]["first_mismatch"] is None
            for case in report["cases"].values()
        )
        report["decision"] = {
            "pallas_matches_materialized_jax": exact_materialized,
            "next_step": "attribute_monolithic_lowering" if exact_materialized else "first_modular_boundary_mismatch",
        }
        report["status"] = "complete"
        checkpoint(result_path, report)
        return report
    except Exception as error:
        report.update(
            status="error", fatal_error_type=type(error).__name__,
            fatal_error=str(error), fatal_traceback=traceback.format_exc(),
        )
        checkpoint(result_path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_boundary_replay(dataset=_dataset_path(args.dataset), output=args.output)
    print("DECISION", json.dumps(report["decision"], allow_nan=False), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
