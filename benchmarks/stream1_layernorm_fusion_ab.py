from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp

from tpu_beam_search.stream1_layernorm_pallas import (
    pallas_fused_dense_layer_norm,
    pallas_layer_norm,
)
from tpu_beam_search.stream1_pallas import pallas_dense_linear


LOCAL_BATCH = 16_384
HIDDEN = 1_024
WARMUPS = 5
REPEATS = 15


def candidate_configs():
    return (
        ("separate", 128, 256, 512),
        ("fused", 128, 256, 512),
        ("separate", 256, 256, 512),
        ("fused", 256, 256, 512),
    )


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
    row = jnp.arange(LOCAL_BATCH, dtype=jnp.float32)[:, None]
    column = jnp.arange(HIDDEN, dtype=jnp.float32)[None, :]
    values = jnp.sin(row * 0.001 + column * 0.01).astype(jnp.bfloat16)
    weight = jnp.sin(
        jnp.arange(HIDDEN * HIDDEN, dtype=jnp.float32).reshape(HIDDEN, HIDDEN)
        * 0.0001
    ).astype(jnp.bfloat16)
    bias = jnp.linspace(-0.2, 0.2, HIDDEN).astype(jnp.bfloat16)
    scale = jnp.linspace(0.8, 1.2, HIDDEN).astype(jnp.bfloat16)
    beta = jnp.linspace(-0.1, 0.1, HIDDEN).astype(jnp.bfloat16)

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "local_batch": LOCAL_BATCH,
            "input_width": HIDDEN,
            "output_width": HIDDEN,
            "dtype": "bfloat16",
            "warmups": WARMUPS,
            "repeats": REPEATS,
        },
        "candidates": [],
    }
    outputs = {}
    for mode, bm, bk, bn in candidate_configs():
        if mode == "separate":
            compiled = jax.jit(
                lambda x, w, b, gamma, ln_beta: pallas_layer_norm(
                    pallas_dense_linear(
                        x, w, b, bm=bm, bk=bk, bn=bn, relu=False
                    ),
                    gamma,
                    ln_beta,
                    bm=bm,
                )
            )
        else:
            compiled = jax.jit(
                lambda x, w, b, gamma, ln_beta: pallas_fused_dense_layer_norm(
                    x,
                    w,
                    b,
                    gamma,
                    ln_beta,
                    bm=bm,
                    bk=bk,
                    bn=bn,
                )
            )
        first, median, samples = measure(
            lambda: compiled(values, weight, bias, scale, beta)
        )
        output = compiled(values, weight, bias, scale, beta)
        outputs[(mode, bm)] = output
        entry = {
            "mode": mode,
            "bm": bm,
            "bk": bk,
            "bn": bn,
            "compile_and_first_seconds": first,
            "steady_seconds_median": median,
            "states_per_second": LOCAL_BATCH / median,
            "finite": bool(jnp.all(jnp.isfinite(output))),
            "samples": samples,
        }
        if mode == "fused":
            separate = outputs[("separate", bm)]
            absolute = jnp.abs(
                output.astype(jnp.float32) - separate.astype(jnp.float32)
            )
            entry["max_abs_vs_separate"] = float(jnp.max(absolute))
            entry["mean_abs_vs_separate"] = float(jnp.mean(absolute))
            separate_entry = next(
                item
                for item in result["candidates"]
                if item["mode"] == "separate" and item["bm"] == bm
            )
            entry["speedup_vs_separate"] = (
                separate_entry["steady_seconds_median"] / median
            )
        result["candidates"].append(entry)
        print("CANDIDATE", json.dumps(entry), flush=True)

    path = Path("/kaggle/working/stream1_layernorm_fusion_ab.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
