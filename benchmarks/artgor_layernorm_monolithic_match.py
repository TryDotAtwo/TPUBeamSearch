"""One-factor JAX/Pallas ladder against monolithic Artgor LayerNorm."""
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
from benchmarks.artgor_layernorm_attribution import (
    attribution_variants,
    jax_layernorm_checkpoints,
    _mapped,
)
from benchmarks.artgor_layernorm_subtraction import CASE_DEFINITIONS, tensor_metrics
from benchmarks.artgor_pallas_exact_diagnostic import _make_states
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import flat_embedding_prepacked
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_pallas_attribution import pallas_layernorm_probe
from tpu_beam_search.stream1_layernorm_pallas_exact import prepare_pallas_exact_weights
from tpu_beam_search.stream1_layernorm_reference import (
    layer_norm_reference,
    layernorm_stream1_weights_from_artgor_params,
)


RESULT_NAME = "artgor_layernorm_monolithic_match.json"
TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 256


def variant_names() -> tuple[str, ...]:
    return tuple(attribution_variants())


def decide_match(cases: Mapping[str, Mapping[str, Mapping[str, Mapping[str, bool]]]]) -> dict:
    if not cases:
        return {"exact_jax_variants": [], "exact_pallas_variants": [], "selected": None}
    names = tuple(next(iter(cases.values())))
    exact_jax = [
        name for name in names
        if all(case[name]["jax"]["exact"] for case in cases.values())
    ]
    exact_pallas = [
        name for name in names
        if all(case[name]["pallas"]["exact"] for case in cases.values())
    ]
    common = [name for name in names if name in exact_jax and name in exact_pallas]
    return {
        "exact_jax_variants": exact_jax,
        "exact_pallas_variants": exact_pallas,
        "selected": common[0] if len(common) == 1 else None,
    }


def run_monolithic_match(*, dataset: Path, output: Path) -> dict:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    variants = attribution_variants()
    report = {
        "schema_version": 1, "status": "running",
        "protocol": {
            "scope": "exact BK128 input Dense and first LayerNorm only; beam excluded",
            "local_batch_per_device": LOCAL_BATCH,
            "variants": list(variants),
            "gate": "zero BF16 mismatches and identical hash against monolithic JAX",
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
        weight_example = (typed.input.normalization.scale, typed.input.normalization.bias)

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
            )), mesh=mesh, weight_example=typed.input.normalization,
        )
        jax_calls = {
            name: _mapped(
                lambda values, weights, arithmetic=arithmetic: jax_layernorm_checkpoints(
                    values, weights[0], weights[1],
                    epsilon=architecture.LAYER_NORM_EPSILON,
                    arithmetic=arithmetic,
                )["relu"],
                mesh=mesh, weight_example=weight_example,
            )
            for name, arithmetic in variants.items()
        }
        pallas_calls = {
            name: _mapped(
                lambda values, weights, arithmetic=arithmetic: pallas_layernorm_probe(
                    values, weights[0], weights[1], checkpoint="relu",
                    epsilon=architecture.LAYER_NORM_EPSILON, bm=128,
                    arithmetic=arithmetic,
                ),
                mesh=mesh, weight_example=weight_example,
            )
            for name, arithmetic in variants.items()
        }
        sample = jax.device_put(
            np.zeros((TARGET_DEVICE_COUNT * LOCAL_BATCH, architecture.HIDDEN1), dtype=np.float32).astype(jnp.bfloat16),
            state_sharding,
        )
        for name, call in pallas_calls.items():
            text = str(call.lower(sample, (scale_d, bias_d)).compiler_ir(dialect="stablehlo"))
            report["hlo_audit"][name] = {
                "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "tpu_custom_call_count": text.count("tpu_custom_call"),
            }
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        global_batch = TARGET_DEVICE_COUNT * LOCAL_BATCH
        for case_name, kind, seed in CASE_DEFINITIONS:
            states_host = _make_states(puzzle, kind, seed, global_batch)
            states_d = jax.device_put(states_host, state_sharding)
            dense = jax.block_until_ready(dense_call(states_d, prepared_d))
            monolithic = jax.block_until_ready(monolithic_call(dense, normalization_d))
            rows = {}
            for name in variants:
                jax_output = jax.block_until_ready(jax_calls[name](dense, (scale_d, bias_d)))
                pallas_output = jax.block_until_ready(pallas_calls[name](dense, (scale_d, bias_d)))
                rows[name] = {
                    "jax": tensor_metrics(monolithic, jax_output),
                    "pallas": tensor_metrics(monolithic, pallas_output),
                    "jax_vs_pallas": tensor_metrics(jax_output, pallas_output),
                }
            report["cases"][case_name] = {
                "kind": kind, "seed": seed, "global_batch": global_batch,
                "input_sha256": hashlib.sha256(states_host.tobytes()).hexdigest(),
                "dense_sha256": hashlib.sha256(np.asarray(dense).tobytes()).hexdigest(),
                "variants": rows,
            }
            checkpoint(result_path, report)
        decision_input = {
            case_name: case["variants"] for case_name, case in report["cases"].items()
        }
        report["decision"] = decide_match(decision_input)
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
    report = run_monolithic_match(dataset=_dataset_path(args.dataset), output=args.output)
    print("DECISION", json.dumps(report["decision"], allow_nan=False), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
