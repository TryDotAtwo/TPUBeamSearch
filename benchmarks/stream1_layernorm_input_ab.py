from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from tpu_beam_search.stream1_architecture import InputEncodingKind, Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas import (
    make_fused_virtual_one_hot_weight,
    pallas_layernorm_input_prefix,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)


LOCAL_BATCH = 16_384
WARMUPS = 5
REPEATS = 15


def candidate_configs():
    return (
        (InputEncodingKind.EMBEDDING_GATHER, 256, 256, 512),
        (InputEncodingKind.EMBEDDING_GATHER, 1024, 256, 512),
        (InputEncodingKind.VIRTUAL_ONE_HOT_MXU, 256, 256, 512),
        (InputEncodingKind.VIRTUAL_ONE_HOT_MXU, 1024, 256, 512),
        (InputEncodingKind.FUSED_VIRTUAL_ONE_HOT, 256, 128, 512),
        (InputEncodingKind.FUSED_VIRTUAL_ONE_HOT, 1024, 128, 512),
    )


def find_dataset() -> Path:
    for path in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("artgor/cube555-tpu-artifacts is not attached")


def original_prefix(params, states, dtype=jnp.bfloat16):
    embedded = params["embed"][states.astype(jnp.int32)]
    hidden = embedded.reshape(embedded.shape[0], -1).astype(dtype)
    layer = params["input_stack"][0]
    hidden = hidden @ layer["lin_w"].astype(dtype) + layer["lin_b"].astype(dtype)
    mean = jnp.mean(hidden, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(hidden - mean), axis=-1, keepdims=True)
    hidden = (
        (hidden - mean)
        * jax.lax.rsqrt(variance + 1e-5)
        * layer["ln_gamma"].astype(dtype)
        + layer["ln_beta"].astype(dtype)
    )
    return jax.nn.relu(hidden)


def measure(call):
    started = time.perf_counter()
    output = call()
    jax.block_until_ready(output)
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
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import load_params_from_pt

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=int(params["state_size"])
    )
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    fused_weight = make_fused_virtual_one_hot_weight(
        weights.embedding,
        weights.input.dense.weight,
        STATE_LEN=architecture.STATE_LEN,
    )
    states = jnp.broadcast_to(
        jnp.arange(architecture.STATE_LEN, dtype=jnp.uint8),
        (LOCAL_BATCH, architecture.STATE_STORAGE_LEN),
    )
    oracle = jax.jit(lambda x: original_prefix(params, x))
    first, median, samples = measure(lambda: oracle(states))
    oracle_output = oracle(states)
    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": "q555_2k_BEST.pt",
            "local_batch": LOCAL_BATCH,
            "state_len": architecture.STATE_LEN,
            "num_classes": architecture.NUM_CLASSES,
            "embed_dim": architecture.EMBED_DIM,
            "hidden": architecture.HIDDEN1,
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
        "candidates": [],
    }
    for encoding, bm, bk_dense, bn_dense in candidate_configs():
        compiled = jax.jit(
            lambda x: pallas_layernorm_input_prefix(
                x,
                weights,
                architecture,
                input_encoding=encoding,
                fused_input_weight=fused_weight,
                bm=bm,
                bk=128,
                bn=128,
                bk_embedding=128,
                bn_embedding=128,
                bk_dense=bk_dense,
                bn_dense=bn_dense,
            )
        )
        first, candidate_median, candidate_samples = measure(lambda: compiled(states))
        output = compiled(states)
        absolute = jnp.abs(
            output.astype(jnp.float32) - oracle_output.astype(jnp.float32)
        )
        entry = {
            "encoding": encoding.value,
            "bm": bm,
            "bk_dense": bk_dense,
            "bn_dense": bn_dense,
            "compile_and_first_seconds": first,
            "steady_seconds_median": candidate_median,
            "states_per_second": LOCAL_BATCH / candidate_median,
            "speedup_vs_original": median / candidate_median,
            "max_abs_vs_original": float(jnp.max(absolute)),
            "mean_abs_vs_original": float(jnp.mean(absolute)),
            "finite": bool(jnp.all(jnp.isfinite(output))),
            "samples": candidate_samples,
        }
        result["candidates"].append(entry)
        print("CANDIDATE", json.dumps(entry), flush=True)

    path = Path("/kaggle/working/stream1_layernorm_input_ab.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
