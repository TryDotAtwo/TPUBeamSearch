from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.stream1_layernorm_pallas import pallas_layer_norm


LOCAL_BATCH = 16_384
HIDDEN = 1_024
EPSILON = 1e-5
WARMUPS = 5
REPEATS = 21


def candidate_bms() -> tuple[int, ...]:
    return (128, 256, 512, 1024)


def original_layer_norm(values, scale, bias):
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (
        (values - mean)
        * jax.lax.rsqrt(variance + EPSILON)
        * scale
        + bias
    )


def fp32_layer_norm(values, scale, bias):
    values = values.astype(jnp.float32)
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (
        (values - mean)
        * jax.lax.rsqrt(variance + EPSILON)
        * scale.astype(jnp.float32)
        + bias.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def measure(call):
    started = time.perf_counter()
    first_output = call()
    jax.block_until_ready(first_output)
    compile_and_first = time.perf_counter() - started
    for _ in range(WARMUPS):
        jax.block_until_ready(call())
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - started)
    return compile_and_first, statistics.median(samples), samples


def main():
    rows = jnp.arange(LOCAL_BATCH, dtype=jnp.float32)[:, None]
    columns = jnp.arange(HIDDEN, dtype=jnp.float32)[None, :]
    values = jnp.sin(rows * 0.001 + columns * 0.01).astype(jnp.bfloat16)
    scale = jnp.linspace(0.5, 1.5, HIDDEN).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.25, 0.25, HIDDEN).astype(jnp.bfloat16)
    fp32_reference = jax.jit(fp32_layer_norm)(values, scale, bias)
    original = jax.jit(original_layer_norm)

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "local_batch": LOCAL_BATCH,
            "hidden": HIDDEN,
            "epsilon": EPSILON,
            "dtype": "bfloat16",
            "reduction_dtype": "float32 for Pallas candidates",
            "warmups": WARMUPS,
            "repeats": REPEATS,
        },
        "original_jax": {},
        "pallas": {},
    }
    first, median, samples = measure(lambda: original(values, scale, bias))
    original_output = original(values, scale, bias)
    result["original_jax"] = {
        "compile_and_first_seconds": first,
        "steady_seconds_median": median,
        "rows_per_second": LOCAL_BATCH / median,
        "samples": samples,
    }

    for bm in candidate_bms():
        candidate = jax.jit(
            lambda x, gamma, beta: pallas_layer_norm(
                x,
                gamma,
                beta,
                bm=bm,
                width_alignment=128,
                epsilon=EPSILON,
            )
        )
        first, median, samples = measure(lambda: candidate(values, scale, bias))
        output = candidate(values, scale, bias)
        absolute = jnp.abs(
            output.astype(jnp.float32) - fp32_reference.astype(jnp.float32)
        )
        original_absolute = jnp.abs(
            output.astype(jnp.float32) - original_output.astype(jnp.float32)
        )
        entry = {
            "bm": bm,
            "compile_and_first_seconds": first,
            "steady_seconds_median": median,
            "rows_per_second": LOCAL_BATCH / median,
            "speedup_vs_original": result["original_jax"]["steady_seconds_median"] / median,
            "max_abs_vs_fp32_reference": float(jnp.max(absolute)),
            "mean_abs_vs_fp32_reference": float(jnp.mean(absolute)),
            "max_abs_vs_original_bf16": float(jnp.max(original_absolute)),
            "finite": bool(jnp.all(jnp.isfinite(output))),
            "samples": samples,
        }
        result["pallas"][str(bm)] = entry
        print("PALLAS", json.dumps(entry), flush=True)

    path = Path("/kaggle/working/stream1_layernorm_tiling.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
