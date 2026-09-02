"""Causal LayerNorm arithmetic attribution for the all-Pallas Artgor engine.

This module deliberately keeps the candidate matrix sequential: every arm
changes exactly one observable rounding boundary relative to ``hlo_mixed``.
The TPU runner uses the same checkpoint names for materialized JAX controls
and Pallas custom calls, so the first divergent value is attributable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from tpu_beam_search.stream1_layernorm_pallas_attribution import (
    PallasLayerNormArithmetic as LayerNormArithmetic,
    pallas_layernorm_probe,
)
from benchmarks.artgor_exact_notebook_validation import _dataset_path, _replicate, checkpoint
from benchmarks.artgor_pallas_exact_diagnostic import _make_states
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_embedding_experimental import flat_embedding_prepacked
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense
from tpu_beam_search.stream1_layernorm_pallas_exact import prepare_pallas_exact_weights
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
RESULT_NAME = "artgor_layernorm_attribution.json"
TARGET_DEVICE_COUNT = 8
LOCAL_BATCH = 256


CHECKPOINT_NAMES = (
    "mean",
    "centered",
    "variance",
    "invstd",
    "affine_fp32",
    "affine_bf16",
    "relu",
)


def attribution_variants() -> dict[str, LayerNormArithmetic]:
    """Return a fixed one-factor ladder, never a Cartesian sweep."""

    baseline = LayerNormArithmetic()
    return {
        "hlo_mixed_control": baseline,
        "fp32_mean": replace(baseline, mean_bf16=False),
        "fp32_variance": replace(baseline, variance_bf16=False),
        "fp32_epsilon": replace(baseline, epsilon_bf16=False),
        "fp32_invstd": replace(baseline, invstd_bf16=False),
        "bf16_affine": replace(baseline, affine_fp32=False),
    }


def jax_layernorm_checkpoints(
    values,
    scale,
    bias,
    *,
    epsilon: float = 1e-5,
    arithmetic: LayerNormArithmetic = LayerNormArithmetic(),
) -> dict[str, jax.Array]:
    """Expose each arithmetic boundary without changing their evaluation order."""

    values_fp32 = values.astype(jnp.float32)
    width = values.shape[-1]
    mean_fp32 = jnp.sum(values_fp32, axis=1, keepdims=True) / width
    mean = mean_fp32.astype(jnp.bfloat16) if arithmetic.mean_bf16 else mean_fp32
    centered = values_fp32 - mean.astype(jnp.float32)
    variance_fp32 = jnp.sum(centered * centered, axis=1, keepdims=True) / width
    variance = (
        variance_fp32.astype(jnp.bfloat16)
        if arithmetic.variance_bf16
        else variance_fp32
    )
    epsilon_value = jnp.asarray(
        epsilon, jnp.bfloat16 if arithmetic.epsilon_bf16 else jnp.float32
    ).astype(jnp.float32)
    invstd_fp32 = jax.lax.rsqrt(variance.astype(jnp.float32) + epsilon_value)
    invstd = (
        invstd_fp32.astype(jnp.bfloat16)
        if arithmetic.invstd_bf16
        else invstd_fp32
    )
    normalized = centered * invstd.astype(jnp.float32)
    if arithmetic.affine_fp32:
        affine_fp32 = (
            normalized * scale.astype(jnp.float32)[None, :]
            + bias.astype(jnp.float32)[None, :]
        )
    else:
        affine_fp32 = (
            normalized.astype(jnp.bfloat16) * scale.astype(jnp.bfloat16)[None, :]
            + bias.astype(jnp.bfloat16)[None, :]
        ).astype(jnp.float32)
    affine_bf16 = affine_fp32.astype(jnp.bfloat16)
    relu = jnp.maximum(affine_bf16, jnp.bfloat16(0)).astype(jnp.bfloat16)
    return {
        "mean": mean,
        "centered": centered,
        "variance": variance,
        "invstd": invstd,
        "affine_fp32": affine_fp32,
        "affine_bf16": affine_bf16,
        "relu": relu,
    }


def _checkpoint_metrics(reference, candidate) -> dict[str, object]:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        return {
            "shape_equal": False,
            "mismatch_count": int(max(reference_array.size, candidate_array.size)),
            "max_abs": None,
            "mean_abs": None,
        }
    equal = np.equal(reference_array, candidate_array)
    finite = np.isfinite(reference_array) & np.isfinite(candidate_array)
    equal = equal | (np.isnan(reference_array) & np.isnan(candidate_array))
    difference = np.abs(
        reference_array.astype(np.float64) - candidate_array.astype(np.float64)
    )
    return {
        "shape_equal": True,
        "finite": bool(np.all(finite)),
        "mismatch_count": int(equal.size - np.count_nonzero(equal)),
        "max_abs": float(np.nanmax(difference)) if difference.size else 0.0,
        "mean_abs": float(np.nanmean(difference)) if difference.size else 0.0,
    }


def compare_checkpoint_sequences(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Compare in reference order and report the first divergent boundary."""

    if tuple(reference) != tuple(candidate):
        raise ValueError("reference and candidate checkpoint order must match")
    metrics = {
        name: _checkpoint_metrics(reference[name], candidate[name])
        for name in reference
    }
    first_mismatch = next(
        (name for name, value in metrics.items() if value["mismatch_count"]),
        None,
    )
    return {"first_mismatch": first_mismatch, "checkpoints": metrics}


def decide_attribution(corpora: Mapping[str, Mapping[str, bool]]) -> dict[str, object]:
    """Promotion gate: exact both in probe and production-shaped dispatches."""

    if not corpora:
        return {"promote": False, "reason": "no_corpora"}
    if not all(result.get("small_exact", False) for result in corpora.values()):
        return {"promote": False, "reason": "small_probe_not_exact"}
    if not all(result.get("production_exact", False) for result in corpora.values()):
        return {"promote": False, "reason": "production_shape_not_exact"}
    return {
        "promote": True,
        "reason": "all_materialized_boundaries_exact",
        "arithmetic": asdict(LayerNormArithmetic()),
    }


def _stablehlo_sha(lowered) -> str:
    text = str(lowered.compiler_ir(dialect="stablehlo"))
    return hashlib.sha256(text.encode()).hexdigest()


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


def _materialized_jax_controls(values, scale, bias, *, mesh, epsilon):
    """J2/J3: each arithmetic boundary is a separately synchronized executable."""
    from jax.sharding import PartitionSpec as P

    def unary(function):
        return jax.jit(jax.shard_map(
            function, mesh=mesh, in_specs=P("core", None),
            out_specs=P("core", None), check_vma=False,
        ))

    def binary(function):
        return jax.jit(jax.shard_map(
            function, mesh=mesh,
            in_specs=(P("core", None), P("core", None)),
            out_specs=P("core", None), check_vma=False,
        ))

    mean_call = unary(lambda x: (
        jnp.sum(x.astype(jnp.float32), axis=1, keepdims=True) / x.shape[1]
    ).astype(jnp.bfloat16))
    centered_call = binary(
        lambda x, mean: x.astype(jnp.float32) - mean.astype(jnp.float32)
    )
    variance_call = unary(lambda centered: (
        jnp.sum(centered * centered, axis=1, keepdims=True) / centered.shape[1]
    ).astype(jnp.bfloat16))
    invstd_call = unary(lambda variance: jax.lax.rsqrt(
        variance.astype(jnp.float32)
        + jnp.asarray(epsilon, jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16))
    affine_call = jax.jit(jax.shard_map(
        lambda centered, invstd, weights: (
            centered * invstd.astype(jnp.float32)
            * weights[0].astype(jnp.float32)[None, :]
            + weights[1].astype(jnp.float32)[None, :]
        ),
        mesh=mesh,
        in_specs=(P("core", None), P("core", None), (P(), P())),
        out_specs=P("core", None),
        check_vma=False,
    ))
    relu_call = unary(
        lambda x: jnp.maximum(x.astype(jnp.bfloat16), jnp.bfloat16(0))
    )
    cast_call = unary(lambda x: x.astype(jnp.bfloat16))
    mean = jax.block_until_ready(mean_call(values))
    centered = jax.block_until_ready(centered_call(values, mean))
    variance = jax.block_until_ready(variance_call(centered))
    invstd = jax.block_until_ready(invstd_call(variance))
    affine_fp32 = jax.block_until_ready(
        affine_call(centered, invstd, (scale, bias))
    )
    affine_bf16 = jax.block_until_ready(cast_call(affine_fp32))
    relu = jax.block_until_ready(relu_call(affine_bf16))
    return {
        "mean": mean,
        "centered": centered,
        "variance": variance,
        "invstd": invstd,
        "affine_fp32": affine_fp32,
        "affine_bf16": affine_bf16,
        "relu": relu,
    }


def run_attribution(*, dataset: Path, output: Path) -> dict:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "scope": "input Dense plus first LayerNorm only; beam excluded",
            "local_batch_per_device": LOCAL_BATCH,
            "controls": ["J0_monolithic", "J1_same_call", "J2_materialized"],
            "candidate_order": list(attribution_variants()),
            "checkpoint_order": list(CHECKPOINT_NAMES),
            "bk1024": "documented negative control from all-Pallas v3; not rerun",
        },
        "context": {},
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
        j0_call = _mapped(
            lambda values, weights: jax.nn.relu(layer_norm_reference(
                values, weights, epsilon=architecture.LAYER_NORM_EPSILON,
            )),
            mesh=mesh,
            weight_example=typed.input.normalization,
        )
        normalization_d = _replicate(typed.input.normalization, mesh)
        j1_calls = {
            name: _mapped(
                lambda values, weights, name=name: jax_layernorm_checkpoints(
                    values, weights[0], weights[1],
                    epsilon=architecture.LAYER_NORM_EPSILON,
                )[name],
                mesh=mesh,
                weight_example=(
                    typed.input.normalization.scale,
                    typed.input.normalization.bias,
                ),
            )
            for name in CHECKPOINT_NAMES
        }
        pallas_calls = {
            variant_name: {
                checkpoint_name: _mapped(
                    lambda values, weights, checkpoint_name=checkpoint_name,
                    arithmetic=arithmetic: pallas_layernorm_probe(
                        values, weights[0], weights[1],
                        checkpoint=checkpoint_name,
                        epsilon=architecture.LAYER_NORM_EPSILON,
                        bm=128,
                        arithmetic=arithmetic,
                    ),
                    mesh=mesh,
                    weight_example=(
                        typed.input.normalization.scale,
                        typed.input.normalization.bias,
                    ),
                )
                for checkpoint_name in CHECKPOINT_NAMES
            }
            for variant_name, arithmetic in attribution_variants().items()
        }
        puzzle = load_puzzle(puzzle_path, state_len=150, move_count=30)
        global_batch = TARGET_DEVICE_COUNT * LOCAL_BATCH
        for case_name, kind, seed in CASE_DEFINITIONS:
            states_host = _make_states(puzzle, kind, seed, global_batch)
            states_d = jax.device_put(states_host, state_sharding)
            dense = jax.block_until_ready(dense_call(states_d, prepared_d))
            j0 = jax.block_until_ready(j0_call(dense, normalization_d))
            j1 = {
                name: jax.block_until_ready(call(dense, (scale_d, bias_d)))
                for name, call in j1_calls.items()
            }
            j2 = _materialized_jax_controls(
                dense, scale_d, bias_d, mesh=mesh,
                epsilon=architecture.LAYER_NORM_EPSILON,
            )
            controls = {
                "j0_vs_j1_final": _checkpoint_metrics(j0, j1["relu"]),
                "j0_vs_j2_final": _checkpoint_metrics(j0, j2["relu"]),
                "j1_vs_j2": compare_checkpoint_sequences(j1, j2),
            }
            candidates = {}
            for variant_name, calls in pallas_calls.items():
                values = {
                    name: jax.block_until_ready(call(dense, (scale_d, bias_d)))
                    for name, call in calls.items()
                }
                reference = {
                    name: (
                        jnp.broadcast_to(j1[name], values[name].shape)
                        if j1[name].shape != values[name].shape
                        else j1[name]
                    )
                    for name in CHECKPOINT_NAMES
                }
                candidates[variant_name] = compare_checkpoint_sequences(
                    reference, values,
                )
            report["cases"][case_name] = {
                "kind": kind,
                "seed": seed,
                "global_batch": global_batch,
                "input_sha256": hashlib.sha256(states_host.tobytes()).hexdigest(),
                "dense_sha256": hashlib.sha256(
                    np.asarray(dense).tobytes()
                ).hexdigest(),
                "logical_shape": list(dense.shape),
                "controls": controls,
                "candidates": candidates,
            }
            checkpoint(result_path, report)
        promotion_inputs = {}
        for case_name, case in report["cases"].items():
            baseline = case["candidates"]["hlo_mixed_control"]
            promotion_inputs[case_name] = {
                "small_exact": baseline["first_mismatch"] is None,
                "production_exact": baseline["first_mismatch"] is None,
            }
        report["decision"] = decide_attribution(promotion_inputs)
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_attribution(
        dataset=_dataset_path(args.dataset), output=args.output,
    )
    print("DECISION", json.dumps(report["decision"], allow_nan=False), flush=True)
    print("RESULT_PATH", args.output / RESULT_NAME, flush=True)


if __name__ == "__main__":
    main()
