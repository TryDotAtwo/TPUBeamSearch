"""Fixed-operand variance-to-invstd and affine TPU attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.artgor_exact_notebook_validation import _dataset_path, _replicate, checkpoint
from benchmarks.artgor_pallas_exact_diagnostic import _make_states
from benchmarks.artgor_layernorm_subtraction import (
    CASE_DEFINITIONS,
    _hlo_identity,
    _mapped,
    _mapped_binary,
    tensor_metrics,
)
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import flat_embedding_prepacked
from tpu_beam_search.stream1_layernorm_invstd import pallas_invstd, pallas_invstd_affine
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_pallas_exact import prepare_pallas_exact_weights
from tpu_beam_search.stream1_layernorm_reference import layernorm_stream1_weights_from_artgor_params


RESULT_NAME = "artgor_layernorm_invstd.json"
TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 256


__all__ = ["RESULT_NAME", "tensor_metrics"]


def _mapped_unary(local_call, *, mesh):
    from jax.sharding import PartitionSpec as P

    return jax.jit(jax.shard_map(
        local_call, mesh=mesh, in_specs=P("core", None),
        out_specs=P("core", None), check_vma=False,
    ))


def run_invstd(*, dataset: Path, output: Path) -> dict:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "scope": "fixed BF16 variance through invstd and one affine control",
            "local_batch_per_device": LOCAL_BATCH,
            "corpora": [name for name, _, _ in CASE_DEFINITIONS],
        },
        "context": {}, "hlo_identity": {}, "cases": {}, "decision": {},
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
        report["context"] = {
            "source_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), text=True,
            ).strip(),
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
        scale_d = _replicate(typed.input.normalization.scale, mesh)
        bias_d = _replicate(typed.input.normalization.bias, mesh)

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
        mean_call = _mapped_unary(lambda values: (
            jnp.sum(values.astype(jnp.float32), axis=1, keepdims=True)
            / values.shape[1]
        ).astype(jnp.bfloat16), mesh=mesh)
        centered_call = _mapped_binary(
            lambda values, mean: values.astype(jnp.float32) - mean.astype(jnp.float32),
            mesh=mesh,
        )
        variance_call = _mapped_unary(lambda centered: (
            jnp.sum(centered * centered, axis=1, keepdims=True) / centered.shape[1]
        ).astype(jnp.bfloat16), mesh=mesh)
        same_fp32 = _mapped_unary(lambda variance: jax.lax.rsqrt(
            variance.astype(jnp.float32)
            + jnp.asarray(architecture.LAYER_NORM_EPSILON, jnp.bfloat16).astype(jnp.float32)
        ), mesh=mesh)
        cast_fp32 = _mapped_unary(lambda value: value.astype(jnp.float32), mesh=mesh)
        add_epsilon = _mapped_unary(lambda variance_fp32: (
            variance_fp32
            + jnp.asarray(architecture.LAYER_NORM_EPSILON, jnp.bfloat16).astype(jnp.float32)
        ), mesh=mesh)
        rsqrt_call = _mapped_unary(jax.lax.rsqrt, mesh=mesh)
        cast_bf16 = _mapped_unary(lambda value: value.astype(jnp.bfloat16), mesh=mesh)
        pallas_fp32 = _mapped_unary(
            lambda variance: pallas_invstd(
                variance, epsilon=architecture.LAYER_NORM_EPSILON,
                output_bf16=False, bm=128,
            ), mesh=mesh,
        )
        pallas_bf16 = _mapped_unary(
            lambda variance: pallas_invstd(
                variance, epsilon=architecture.LAYER_NORM_EPSILON,
                output_bf16=True, bm=128,
            ), mesh=mesh,
        )
        interpret_fp32 = _mapped_unary(
            lambda variance: pallas_invstd(
                variance, epsilon=architecture.LAYER_NORM_EPSILON,
                output_bf16=False, bm=128, interpret=True,
            ), mesh=mesh,
        )
        affine_jax = jax.jit(jax.shard_map(
            lambda centered, invstd, weights: (
                centered * invstd.astype(jnp.float32)
                * weights[0].astype(jnp.float32)[None, :]
                + weights[1].astype(jnp.float32)[None, :]
            ).astype(jnp.bfloat16),
            mesh=mesh,
            in_specs=(P("core", None), P("core", None), (P(), P())),
            out_specs=P("core", None), check_vma=False,
        ))
        affine_pallas = jax.jit(jax.shard_map(
            lambda centered, invstd, weights: pallas_invstd_affine(
                centered, invstd, weights[0], weights[1], bm=128,
            ),
            mesh=mesh,
            in_specs=(P("core", None), P("core", None), (P(), P())),
            out_specs=P("core", None), check_vma=False,
        ))
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        global_batch = TARGET_DEVICE_COUNT * LOCAL_BATCH
        for case_index, (case_name, kind, seed) in enumerate(CASE_DEFINITIONS):
            states_host = _make_states(puzzle, kind, seed, global_batch)
            states_d = jax.device_put(states_host, state_sharding)
            dense = jax.block_until_ready(dense_call(states_d, prepared_d))
            mean = jax.block_until_ready(mean_call(dense))
            centered = jax.block_until_ready(centered_call(dense, mean))
            variance = jax.block_until_ready(variance_call(centered))
            jax_same_fp32 = jax.block_until_ready(same_fp32(variance))
            variance_fp32 = jax.block_until_ready(cast_fp32(variance))
            added = jax.block_until_ready(add_epsilon(variance_fp32))
            jax_materialized_fp32 = jax.block_until_ready(rsqrt_call(added))
            jax_materialized_bf16 = jax.block_until_ready(
                cast_bf16(jax_materialized_fp32)
            )
            real_fp32 = jax.block_until_ready(pallas_fp32(variance))
            real_bf16 = jax.block_until_ready(pallas_bf16(variance))
            interpreted_fp32 = jax.block_until_ready(interpret_fp32(variance))
            expected_affine = jax.block_until_ready(
                affine_jax(centered, jax_materialized_bf16, (scale_d, bias_d))
            )
            actual_affine = jax.block_until_ready(
                affine_pallas(centered, jax_materialized_bf16, (scale_d, bias_d))
            )
            if case_index == 0:
                report["hlo_identity"] = {
                    "jax_same_fp32": _hlo_identity(same_fp32, variance),
                    "jax_materialized_rsqrt": _hlo_identity(rsqrt_call, added),
                    "pallas_fp32": _hlo_identity(pallas_fp32, variance),
                    "pallas_bf16": _hlo_identity(pallas_bf16, variance),
                    "pallas_interpret_fp32": _hlo_identity(interpret_fp32, variance),
                    "pallas_affine": _hlo_identity(
                        affine_pallas, centered, jax_materialized_bf16,
                        (scale_d, bias_d),
                    ),
                }
            report["cases"][case_name] = {
                "kind": kind, "seed": seed, "global_batch": global_batch,
                "input_sha256": hashlib.sha256(states_host.tobytes()).hexdigest(),
                "variance_sha256": hashlib.sha256(np.asarray(variance).tobytes()).hexdigest(),
                "centered_sha256": hashlib.sha256(np.asarray(centered).tobytes()).hexdigest(),
                "invstd_fp32": {
                    "same_vs_materialized": tensor_metrics(
                        jax_same_fp32, jax_materialized_fp32,
                    ),
                    "materialized_vs_pallas_real": tensor_metrics(
                        jax_materialized_fp32, real_fp32,
                    ),
                    "materialized_vs_pallas_interpret": tensor_metrics(
                        jax_materialized_fp32, interpreted_fp32,
                    ),
                    "interpret_vs_real": tensor_metrics(interpreted_fp32, real_fp32),
                },
                "invstd_bf16": {
                    "materialized_vs_pallas_real": tensor_metrics(
                        jax_materialized_bf16, real_bf16,
                    ),
                },
                "affine_bf16": {
                    "materialized_jax_vs_pallas": tensor_metrics(
                        expected_affine, actual_affine,
                    ),
                },
            }
            checkpoint(result_path, report)
        exact_invstd = all(
            case["invstd_bf16"]["materialized_vs_pallas_real"]["exact"]
            for case in report["cases"].values()
        )
        exact_affine = all(
            case["affine_bf16"]["materialized_jax_vs_pallas"]["exact"]
            for case in report["cases"].values()
        )
        report["decision"] = {
            "invstd_exact": exact_invstd,
            "affine_exact": exact_affine,
            "next": (
                "integrate_materialized_invstd_and_affine"
                if exact_invstd and exact_affine
                else "isolate_first_nonexact_boundary"
            ),
        }
        report["status"] = "complete" if exact_invstd and exact_affine else "rejected"
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
    report = run_invstd(dataset=_dataset_path(args.dataset), output=args.output)
    print("DECISION", json.dumps(report["decision"], allow_nan=False), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
