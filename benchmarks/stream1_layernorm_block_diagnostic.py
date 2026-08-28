from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from benchmarks.stream1_layernorm_full_mlp import make_valid_states
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas import (
    pallas_fused_dense_layer_norm,
    pallas_fused_residual_block,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)
from tpu_beam_search.stream1_pallas import pallas_dense_linear


LOCAL_BATCH = 16_384
WARMUPS = 3
REPEATS = 9


def diagnostic_levels():
    return ("dense1", "dense1_layernorm_relu", "residual_block")


def residual_block_candidates():
    return ("two_kernel", "one_kernel")


def find_dataset() -> Path:
    for path in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("artgor/cube555-tpu-artifacts is not attached")


def layer_norm_bf16(x, gamma, beta, epsilon=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * gamma + beta


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


def metrics(actual, expected):
    absolute = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
    return {
        "finite": bool(jnp.all(jnp.isfinite(actual))),
        "max_abs": float(jnp.max(absolute)),
        "mean_abs": float(jnp.mean(absolute)),
        "exact_fraction": float(jnp.mean(actual == expected)),
    }


def main():
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import load_params_from_pt

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=int(params["state_size"])
    )
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    states = make_valid_states(
        LOCAL_BATCH, architecture.STATE_LEN, architecture.NUM_CLASSES
    )

    def prefix(x):
        embedding = params["embed"][x.astype(jnp.int32)]
        hidden = embedding.reshape(x.shape[0], -1).astype(jnp.bfloat16)
        layer = params["input_stack"][0]
        hidden = hidden @ layer["lin_w"].astype(jnp.bfloat16)
        hidden += layer["lin_b"].astype(jnp.bfloat16)
        hidden = layer_norm_bf16(
            hidden,
            layer["ln_gamma"].astype(jnp.bfloat16),
            layer["ln_beta"].astype(jnp.bfloat16),
        )
        return jax.nn.relu(hidden).astype(jnp.bfloat16)

    hidden = jax.jit(prefix)(states)
    block = weights.residuals[0]

    def jax_dense1(x):
        return (
            x @ block.first.dense.weight.astype(jnp.bfloat16)
            + block.first.dense.bias.astype(jnp.bfloat16)
        ).astype(jnp.bfloat16)

    def jax_dense1_ln_relu(x):
        dense = jax_dense1(x)
        normalized = layer_norm_bf16(
            dense,
            block.first.normalization.scale.astype(jnp.bfloat16),
            block.first.normalization.bias.astype(jnp.bfloat16),
        )
        return jax.nn.relu(normalized).astype(jnp.bfloat16)

    def jax_block(x):
        branch = jax_dense1_ln_relu(x)
        branch = (
            branch @ block.second.dense.weight.astype(jnp.bfloat16)
            + block.second.dense.bias.astype(jnp.bfloat16)
        ).astype(jnp.bfloat16)
        branch = layer_norm_bf16(
            branch,
            block.second.normalization.scale.astype(jnp.bfloat16),
            block.second.normalization.bias.astype(jnp.bfloat16),
        )
        return jax.nn.relu(x + branch).astype(jnp.bfloat16)

    def pallas_dense1(x):
        return pallas_dense_linear(
            x,
            block.first.dense.weight,
            block.first.dense.bias,
            bm=256,
            bk=256,
            bn=512,
            relu=False,
        )

    def pallas_dense1_ln_relu(x):
        return pallas_fused_dense_layer_norm(
            x,
            block.first.dense.weight,
            block.first.dense.bias,
            block.first.normalization.scale,
            block.first.normalization.bias,
            relu=True,
            bm=256,
            bk=256,
            bn=512,
            fp32_statistics=False,
        )

    def pallas_block(x):
        branch = pallas_dense1_ln_relu(x)
        return pallas_fused_dense_layer_norm(
            branch,
            block.second.dense.weight,
            block.second.dense.bias,
            block.second.normalization.scale,
            block.second.normalization.bias,
            skip=x,
            add_skip=True,
            relu=True,
            bm=256,
            bk=256,
            bn=512,
            fp32_statistics=False,
        )

    def pallas_block_one_kernel(x):
        return pallas_fused_residual_block(
            x,
            block,
            bm=256,
            bk=256,
            bn=512,
            fp32_statistics=False,
        )

    pairs = {
        "dense1": (jax.jit(jax_dense1), jax.jit(pallas_dense1)),
        "dense1_layernorm_relu": (
            jax.jit(jax_dense1_ln_relu),
            jax.jit(pallas_dense1_ln_relu),
        ),
        "residual_block": (jax.jit(jax_block), jax.jit(pallas_block)),
    }
    result = {
        "contract": {
            "checkpoint": "q555_2k_BEST.pt",
            "batch": LOCAL_BATCH,
            "hidden": architecture.HIDDEN1,
            "dtype": "bfloat16",
            "same_input_for_all_levels": True,
            "input_source": "original JAX input prefix on valid diverse states",
            "bm": 256,
            "bk": 256,
            "bn": 512,
            "warmups": WARMUPS,
            "repeats": REPEATS,
        },
        "levels": {},
    }
    for name in diagnostic_levels():
        reference_call, pallas_call = pairs[name]
        reference_first, reference_steady, reference_samples = measure(
            lambda: reference_call(hidden)
        )
        pallas_first, pallas_steady, pallas_samples = measure(
            lambda: pallas_call(hidden)
        )
        expected = reference_call(hidden)
        actual = pallas_call(hidden)
        result["levels"][name] = {
            "jax": {
                "compile_and_first_seconds": reference_first,
                "steady_seconds_median": reference_steady,
                "states_per_second": LOCAL_BATCH / reference_steady,
                "samples": reference_samples,
            },
            "pallas": {
                "compile_and_first_seconds": pallas_first,
                "steady_seconds_median": pallas_steady,
                "states_per_second": LOCAL_BATCH / pallas_steady,
                "speedup_vs_jax": reference_steady / pallas_steady,
                "samples": pallas_samples,
            },
            "correctness": metrics(actual, expected),
        }
        if name == "residual_block":
            one_kernel_call = jax.jit(pallas_block_one_kernel)
            one_first, one_steady, one_samples = measure(
                lambda: one_kernel_call(hidden)
            )
            one_output = one_kernel_call(hidden)
            result["levels"][name]["pallas_two_kernel"] = result[
                "levels"
            ][name].pop("pallas")
            result["levels"][name]["two_kernel_correctness"] = result[
                "levels"
            ][name].pop("correctness")
            result["levels"][name]["pallas_one_kernel"] = {
                "compile_and_first_seconds": one_first,
                "steady_seconds_median": one_steady,
                "states_per_second": LOCAL_BATCH / one_steady,
                "speedup_vs_jax": reference_steady / one_steady,
                "speedup_vs_two_kernel": pallas_steady / one_steady,
                "samples": one_samples,
            }
            result["levels"][name]["one_kernel_correctness"] = metrics(
                one_output, expected
            )
        print(name, json.dumps(result["levels"][name]), flush=True)

    path = Path("/kaggle/working/stream1_layernorm_block_diagnostic.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
