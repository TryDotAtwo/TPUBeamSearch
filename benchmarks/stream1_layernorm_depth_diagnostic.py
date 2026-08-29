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


BATCH = 8192
REPEATS = 3
RESULT_PATH = Path("/kaggle/working/stream1_layernorm_depth_diagnostic.json")


def diagnostic_depths(residual_count: int):
    return tuple(range(1, residual_count + 1))


def depth_configs():
    return (
        {
            "id": "per_block-bm128-bk256-bn512-statsbf16",
            "fusion": "per_block",
            "bm": 128,
            "bk": 256,
            "bn": 512,
            "fp32_statistics": False,
        },
        {
            "id": "per_block-bm128-bk256-bn512-statsfp32",
            "fusion": "per_block",
            "bm": 128,
            "bk": 256,
            "bn": 512,
            "fp32_statistics": True,
        },
        {
            "id": "per_layer-bm256-bk256-bn512-statsbf16",
            "fusion": "per_layer",
            "bm": 256,
            "bk": 256,
            "bn": 512,
            "fp32_statistics": False,
        },
        {
            "id": "per_layer-bm256-bk256-bn512-statsfp32",
            "fusion": "per_layer",
            "bm": 256,
            "bk": 256,
            "bn": 512,
            "fp32_statistics": True,
        },
    )


def begin_config(result, config):
    config_result = {**config, "depths": []}
    result["configs"].append(config_result)
    return config_result


def find_dataset() -> Path:
    for path in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("artgor/cube555-tpu-artifacts is not attached")


def checkpoint(result):
    temporary = RESULT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(RESULT_PATH)


def layer_norm(x, scale, bias, epsilon):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


def hidden_metrics(actual, expected):
    actual_fp32 = actual.astype(jnp.float32)
    expected_fp32 = expected.astype(jnp.float32)
    difference = actual_fp32 - expected_fp32
    absolute = jnp.abs(difference)
    dot = jnp.sum(actual_fp32 * expected_fp32, axis=-1)
    norms = jnp.linalg.norm(actual_fp32, axis=-1) * jnp.linalg.norm(
        expected_fp32, axis=-1
    )
    return {
        "finite": bool(jnp.all(jnp.isfinite(actual))),
        "max_abs": float(jnp.max(absolute)),
        "mean_abs": float(jnp.mean(absolute)),
        "rmse": float(jnp.sqrt(jnp.mean(jnp.square(difference)))),
        "mean_cosine": float(jnp.mean(dot / jnp.maximum(norms, 1e-12))),
        "exact_fraction": float(jnp.mean(actual == expected)),
    }


def output_metrics(actual, expected):
    metrics = hidden_metrics(actual, expected)
    metrics["argmax_agreement"] = float(
        jnp.mean(jnp.argmax(actual, axis=-1) == jnp.argmax(expected, axis=-1))
    )
    return metrics


def timed(call, value):
    started = time.perf_counter()
    output = call(value)
    jax.block_until_ready(output)
    compile_and_first = time.perf_counter() - started
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(call(value))
        samples.append(time.perf_counter() - started)
    return output, compile_and_first, statistics.median(samples), samples


def main():
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import load_params_from_pt, num_params

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=int(params["state_size"])
    )
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    epsilon = architecture.LAYER_NORM_EPSILON

    def reference_prefix(states):
        hidden = weights.embedding[states.astype(jnp.int32)].reshape(
            states.shape[0], architecture.STATE_LEN * architecture.EMBED_DIM
        )
        hidden = hidden @ weights.input.dense.weight + weights.input.dense.bias
        hidden = layer_norm(
            hidden,
            weights.input.normalization.scale,
            weights.input.normalization.bias,
            epsilon,
        )
        return jax.nn.relu(hidden).astype(jnp.bfloat16)

    def reference_block(hidden, block):
        branch = hidden @ block.first.dense.weight + block.first.dense.bias
        branch = layer_norm(
            branch,
            block.first.normalization.scale,
            block.first.normalization.bias,
            epsilon,
        )
        branch = jax.nn.relu(branch).astype(jnp.bfloat16)
        branch = branch @ block.second.dense.weight + block.second.dense.bias
        branch = layer_norm(
            branch,
            block.second.normalization.scale,
            block.second.normalization.bias,
            epsilon,
        )
        return jax.nn.relu(hidden + branch).astype(jnp.bfloat16)

    def reference_head(hidden):
        return (
            hidden @ weights.output.weight + weights.output.bias
        ).astype(jnp.bfloat16)

    states = make_valid_states(
        BATCH, architecture.STATE_LEN, architecture.NUM_CLASSES
    )
    reference_hiddens = [jax.jit(reference_prefix)(states)]
    for block in weights.residuals:
        call = jax.jit(lambda x, current=block: reference_block(x, current))
        reference_hiddens.append(call(reference_hiddens[-1]))
    oracle_output = jax.jit(reference_head)(reference_hiddens[-1])

    suffix_calls = {}
    for depth in diagnostic_depths(architecture.RESIDUAL_COUNT):
        def suffix(hidden, start=depth):
            for block in weights.residuals[start:]:
                hidden = reference_block(hidden, block)
            return reference_head(hidden)

        suffix_calls[depth] = jax.jit(suffix)

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": "q555_2k_BEST.pt",
            "parameters": num_params(params),
            "batch": BATCH,
            "state_len": architecture.STATE_LEN,
            "num_classes": architecture.NUM_CLASSES,
            "embed_dim": architecture.EMBED_DIM,
            "hidden": architecture.HIDDEN1,
            "residual_count": architecture.RESIDUAL_COUNT,
            "move_count": architecture.MOVE_COUNT,
            "dtype": "bfloat16",
            "repeats": REPEATS,
            "prefix_input": "shared original JAX input prefix",
        },
        "configs": [],
    }
    checkpoint(result)

    for config in depth_configs():
        config_result = begin_config(result, config)
        checkpoint(result)
        cumulative_hidden = reference_hiddens[0]
        try:
            block_calls = []
            for block in weights.residuals:
                if config["fusion"] == "per_block":
                    call = jax.jit(
                        lambda x, current=block, c=config: pallas_fused_residual_block(
                            x,
                            current,
                            bm=c["bm"],
                            bk=c["bk"],
                            bn=c["bn"],
                            epsilon=epsilon,
                            fp32_statistics=c["fp32_statistics"],
                        )
                    )
                else:
                    def two_kernel(x, current=block, c=config):
                        branch = pallas_fused_dense_layer_norm(
                            x,
                            current.first.dense.weight,
                            current.first.dense.bias,
                            current.first.normalization.scale,
                            current.first.normalization.bias,
                            relu=True,
                            bm=c["bm"], bk=c["bk"], bn=c["bn"],
                            fp32_statistics=c["fp32_statistics"],
                        )
                        return pallas_fused_dense_layer_norm(
                            branch,
                            current.second.dense.weight,
                            current.second.dense.bias,
                            current.second.normalization.scale,
                            current.second.normalization.bias,
                            skip=x, add_skip=True, relu=True,
                            bm=c["bm"], bk=c["bk"], bn=c["bn"],
                            fp32_statistics=c["fp32_statistics"],
                        )

                    call = jax.jit(two_kernel)
                block_calls.append(call)

            for depth, call in zip(
                diagnostic_depths(architecture.RESIDUAL_COUNT), block_calls
            ):
                isolated_output, isolated_first, isolated_steady, samples = timed(
                    call, reference_hiddens[depth - 1]
                )
                cumulative_output, cumulative_first, cumulative_steady, _ = timed(
                    call, cumulative_hidden
                )
                cumulative_hidden = cumulative_output
                hybrid_output = suffix_calls[depth](cumulative_hidden)
                depth_entry = {
                    "depth": depth,
                    "isolated_hidden": hidden_metrics(
                        isolated_output, reference_hiddens[depth]
                    ),
                    "cumulative_hidden": hidden_metrics(
                        cumulative_hidden, reference_hiddens[depth]
                    ),
                    "hybrid_final_output": output_metrics(
                        hybrid_output, oracle_output
                    ),
                    "isolated_compile_and_first_seconds": isolated_first,
                    "isolated_steady_seconds_median": isolated_steady,
                    "cumulative_compile_and_first_seconds": cumulative_first,
                    "cumulative_steady_seconds_median": cumulative_steady,
                    "samples": samples,
                }
                config_result["depths"].append(depth_entry)
                checkpoint(result)
                print(config["id"], depth, json.dumps(depth_entry), flush=True)
            config_result["status"] = "complete"
        except Exception as error:
            config_result["status"] = "rejected_error"
            config_result["error_type"] = type(error).__name__
            config_result["error"] = str(error)
        checkpoint(result)

    print("RESULT_PATH", RESULT_PATH, flush=True)


if __name__ == "__main__":
    main()
