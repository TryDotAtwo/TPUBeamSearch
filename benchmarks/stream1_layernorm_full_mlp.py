from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from tpu_beam_search.stream1_architecture import InputEncodingKind, Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas import stream1_layernorm_pallas_inference
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)


LOCAL_BATCH = 16_384
WARMUPS = 3
REPEATS = 9


def candidate_configs():
    return (
        ("separate", False, 256, 256, 512),
        ("per_layer", False, 256, 256, 512),
    )


def find_dataset() -> Path:
    for path in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("artgor/cube555-tpu-artifacts is not attached")


def measure(call):
    started = time.perf_counter()
    output = call()
    jax.block_until_ready(output)
    first = time.perf_counter() - started
    for _ in range(WARMUPS):
        jax.block_until_ready(call())
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - started)
    return first, statistics.median(samples), samples


def main():
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import apply as original_apply, load_params_from_pt, num_params

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=int(params["state_size"])
    )
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    states = jnp.broadcast_to(
        jnp.arange(architecture.STATE_LEN, dtype=jnp.uint8),
        (LOCAL_BATCH, architecture.STATE_STORAGE_LEN),
    )
    original = jax.jit(
        lambda x: original_apply(params, x, dtype=jnp.bfloat16)
    )
    first, median, samples = measure(lambda: original(states))
    oracle = original(states)
    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": "q555_2k_BEST.pt",
            "parameters": num_params(params),
            "local_batch": LOCAL_BATCH,
            "state_len": architecture.STATE_LEN,
            "hidden": architecture.HIDDEN1,
            "residual_count": architecture.RESIDUAL_COUNT,
            "move_count": architecture.MOVE_COUNT,
            "input_encoding": InputEncodingKind.EMBEDDING_GATHER.value,
            "dtype": "bfloat16",
            "warmups": WARMUPS,
            "repeats": REPEATS,
        },
        "original_jax": {
            "compile_and_first_seconds": first,
            "steady_seconds_median": median,
            "states_per_second": LOCAL_BATCH / median,
            "samples": samples,
        },
        "pallas": [],
    }
    path = Path("/kaggle/working/stream1_layernorm_full_mlp.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for fusion, fp32_statistics, bm, bk, bn in candidate_configs():
        compiled = jax.jit(
            lambda x: stream1_layernorm_pallas_inference(
                x,
                weights,
                architecture,
                input_encoding=InputEncodingKind.EMBEDDING_GATHER,
                bm=bm,
                bk_input=bk,
                bn_input=bn,
                bk_hidden=bk,
                bn_hidden=bn,
                bk_output=256,
                bn_output=128,
                layernorm_fusion=fusion,
                fp32_statistics=fp32_statistics,
            )
        )
        try:
            candidate_first, candidate_median, candidate_samples = measure(
                lambda: compiled(states)
            )
            output = compiled(states)
            absolute = jnp.abs(
                output.astype(jnp.float32) - oracle.astype(jnp.float32)
            )
            finite = bool(jnp.all(jnp.isfinite(output)))
            max_abs = float(jnp.max(absolute))
            mean_abs = float(jnp.mean(absolute))
            argmax_agreement = float(
                jnp.mean(
                    jnp.argmax(output, axis=-1)
                    == jnp.argmax(oracle, axis=-1)
                )
            )
            correctness_valid = finite and max_abs == 0.0 and argmax_agreement == 1.0
            entry = {
                "fusion": fusion,
                "statistics_dtype": "float32" if fp32_statistics else "bfloat16",
                "bm": bm,
                "bk": bk,
                "bn": bn,
                "status": "valid" if correctness_valid else "correctness_failed",
                "compile_and_first_seconds": candidate_first,
                "steady_seconds_median": candidate_median,
                "states_per_second": LOCAL_BATCH / candidate_median,
                "speedup_vs_original": median / candidate_median,
                "max_abs_vs_original": max_abs,
                "mean_abs_vs_original": mean_abs,
                "argmax_agreement": argmax_agreement,
                "finite": finite,
                "samples": candidate_samples,
            }
        except Exception as error:
            entry = {
                "fusion": fusion,
                "statistics_dtype": "float32" if fp32_statistics else "bfloat16",
                "bm": bm,
                "bk": bk,
                "bn": bn,
                "status": "rejected_compile_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        result["pallas"].append(entry)
        print("PALLAS", json.dumps(entry), flush=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
