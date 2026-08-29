from __future__ import annotations

import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from benchmarks.stream1_layernorm_full_mlp import make_valid_states
from tpu_beam_search.stream1_architecture import InputEncodingKind, Stream1Architecture
from tpu_beam_search.stream1_layernorm_pallas import (
    pallas_fused_dense_layer_norm,
    pallas_fused_residual_block,
    stream1_layernorm_pallas_inference,
)
from tpu_beam_search.stream1_layernorm_reference import (
    layernorm_stream1_weights_from_artgor_params,
)


SCREEN_BATCH = 4096
PROMOTION_BATCHES = (16_384, 32_768)
FULL_BATCH = 16_384
RESULT_PATH = Path("/kaggle/working/stream1_layernorm_comprehensive.json")


def screening_configs():
    configs = []
    for bm, bk, bn, fp32_statistics, fusion in itertools.product(
        (128, 256),
        (128, 256),
        (256, 512),
        (False, True),
        ("per_layer", "per_block"),
    ):
        configs.append(
            {
                "id": f"{fusion}-bm{bm}-bk{bk}-bn{bn}-stats{'fp32' if fp32_statistics else 'bf16'}",
                "fusion": fusion,
                "bm": bm,
                "bk": bk,
                "bn": bn,
                "fp32_statistics": fp32_statistics,
            }
        )
    return tuple(configs)


def select_fastest_valid(entries, count):
    valid = [entry for entry in entries if entry.get("status") == "valid"]
    return sorted(valid, key=lambda entry: entry["states_per_second"], reverse=True)[
        :count
    ]


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


def measure(call, *, warmups, repeats):
    started = time.perf_counter()
    output = call()
    jax.block_until_ready(output)
    first = time.perf_counter() - started
    for _ in range(warmups):
        jax.block_until_ready(call())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - started)
    return first, statistics.median(samples), samples


def correctness(actual, expected, *, output_head=False):
    absolute = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
    finite = bool(jnp.all(jnp.isfinite(actual)))
    max_abs = float(jnp.max(absolute))
    mean_abs = float(jnp.mean(absolute))
    entry = {
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "exact_fraction": float(jnp.mean(actual == expected)),
    }
    if output_head:
        agreement = float(
            jnp.mean(
                jnp.argmax(actual, axis=-1) == jnp.argmax(expected, axis=-1)
            )
        )
        entry["argmax_agreement"] = agreement
        entry["valid"] = finite and max_abs <= 0.5 and agreement >= 0.99
    else:
        entry["valid"] = finite and max_abs <= 0.5 and mean_abs <= 0.005
    return entry


def jax_layer_norm(x, scale, bias, epsilon):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * scale + bias


def main():
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import apply as original_apply, load_params_from_pt, num_params

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    architecture = Stream1Architecture.from_artgor_params(
        params, STATE_STORAGE_LEN=int(params["state_size"])
    )
    weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
    block = weights.residuals[0]

    def original_prefix(states):
        embedded = params["embed"][states.astype(jnp.int32)]
        hidden = embedded.reshape(states.shape[0], -1).astype(jnp.bfloat16)
        layer = params["input_stack"][0]
        hidden = hidden @ layer["lin_w"].astype(jnp.bfloat16)
        hidden += layer["lin_b"].astype(jnp.bfloat16)
        hidden = jax_layer_norm(
            hidden,
            layer["ln_gamma"].astype(jnp.bfloat16),
            layer["ln_beta"].astype(jnp.bfloat16),
            architecture.LAYER_NORM_EPSILON,
        )
        return jax.nn.relu(hidden).astype(jnp.bfloat16)

    def original_block(hidden):
        first = hidden @ block.first.dense.weight + block.first.dense.bias
        first = jax_layer_norm(
            first,
            block.first.normalization.scale,
            block.first.normalization.bias,
            architecture.LAYER_NORM_EPSILON,
        )
        first = jax.nn.relu(first).astype(jnp.bfloat16)
        second = first @ block.second.dense.weight + block.second.dense.bias
        second = jax_layer_norm(
            second,
            block.second.normalization.scale,
            block.second.normalization.bias,
            architecture.LAYER_NORM_EPSILON,
        )
        return jax.nn.relu(hidden + second).astype(jnp.bfloat16)

    prefix_call = jax.jit(original_prefix)
    block_oracle_call = jax.jit(original_block)
    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": "q555_2k_BEST.pt",
            "parameters": num_params(params),
            "state_len": architecture.STATE_LEN,
            "num_classes": architecture.NUM_CLASSES,
            "embed_dim": architecture.EMBED_DIM,
            "hidden": architecture.HIDDEN1,
            "residual_count": architecture.RESIDUAL_COUNT,
            "move_count": architecture.MOVE_COUNT,
            "dtype": "bfloat16",
            "screen_batch": SCREEN_BATCH,
            "promotion_batches": PROMOTION_BATCHES,
            "full_batch": FULL_BATCH,
        },
        "screening": [],
        "promoted": [],
        "full_model": [],
        "scaling": {"status": "not_started"},
    }
    checkpoint(result)

    screen_states = make_valid_states(
        SCREEN_BATCH, architecture.STATE_LEN, architecture.NUM_CLASSES
    )
    screen_hidden = prefix_call(screen_states)
    screen_oracle = block_oracle_call(screen_hidden)

    for config in screening_configs():
        try:
            if config["fusion"] == "per_block":
                candidate = jax.jit(
                    lambda x, c=config: pallas_fused_residual_block(
                        x,
                        block,
                        bm=c["bm"],
                        bk=c["bk"],
                        bn=c["bn"],
                        epsilon=architecture.LAYER_NORM_EPSILON,
                        fp32_statistics=c["fp32_statistics"],
                    )
                )
            else:
                def two_kernel(x, c=config):
                    first = pallas_fused_dense_layer_norm(
                        x,
                        block.first.dense.weight,
                        block.first.dense.bias,
                        block.first.normalization.scale,
                        block.first.normalization.bias,
                        relu=True,
                        bm=c["bm"], bk=c["bk"], bn=c["bn"],
                        fp32_statistics=c["fp32_statistics"],
                    )
                    return pallas_fused_dense_layer_norm(
                        first,
                        block.second.dense.weight,
                        block.second.dense.bias,
                        block.second.normalization.scale,
                        block.second.normalization.bias,
                        skip=x, add_skip=True, relu=True,
                        bm=c["bm"], bk=c["bk"], bn=c["bn"],
                        fp32_statistics=c["fp32_statistics"],
                    )
                candidate = jax.jit(two_kernel)
            first, steady, samples = measure(
                lambda: candidate(screen_hidden), warmups=2, repeats=5
            )
            output = candidate(screen_hidden)
            check = correctness(output, screen_oracle)
            entry = {
                **config,
                "status": "valid" if check["valid"] else "correctness_failed",
                "compile_and_first_seconds": first,
                "steady_seconds_median": steady,
                "states_per_second": SCREEN_BATCH / steady,
                "correctness": check,
                "samples": samples,
            }
        except Exception as error:
            entry = {
                **config,
                "status": "rejected_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        result["screening"].append(entry)
        checkpoint(result)
        print("SCREEN", json.dumps(entry), flush=True)

    promoted = select_fastest_valid(result["screening"], 3)
    for promoted_entry in promoted:
        config = {key: promoted_entry[key] for key in (
            "id", "fusion", "bm", "bk", "bn", "fp32_statistics"
        )}
        promotion_result = {**config, "batches": []}
        for batch in PROMOTION_BATCHES:
            states = make_valid_states(
                batch, architecture.STATE_LEN, architecture.NUM_CLASSES
            )
            hidden = prefix_call(states)
            oracle = block_oracle_call(hidden)
            try:
                if config["fusion"] == "per_block":
                    call = jax.jit(lambda x, c=config: pallas_fused_residual_block(
                        x, block, bm=c["bm"], bk=c["bk"], bn=c["bn"],
                        epsilon=architecture.LAYER_NORM_EPSILON,
                        fp32_statistics=c["fp32_statistics"],
                    ))
                else:
                    def promoted_two_kernel(x, c=config):
                        first = pallas_fused_dense_layer_norm(
                            x, block.first.dense.weight, block.first.dense.bias,
                            block.first.normalization.scale,
                            block.first.normalization.bias, relu=True,
                            bm=c["bm"], bk=c["bk"], bn=c["bn"],
                            fp32_statistics=c["fp32_statistics"],
                        )
                        return pallas_fused_dense_layer_norm(
                            first, block.second.dense.weight, block.second.dense.bias,
                            block.second.normalization.scale,
                            block.second.normalization.bias, skip=x,
                            add_skip=True, relu=True,
                            bm=c["bm"], bk=c["bk"], bn=c["bn"],
                            fp32_statistics=c["fp32_statistics"],
                        )
                    call = jax.jit(promoted_two_kernel)
                first, steady, samples = measure(
                    lambda: call(hidden), warmups=3, repeats=7
                )
                check = correctness(call(hidden), oracle)
                batch_entry = {
                    "batch": batch,
                    "status": "valid" if check["valid"] else "correctness_failed",
                    "compile_and_first_seconds": first,
                    "steady_seconds_median": steady,
                    "states_per_second": batch / steady,
                    "correctness": check,
                    "samples": samples,
                }
            except Exception as error:
                batch_entry = {
                    "batch": batch,
                    "status": "rejected_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            promotion_result["batches"].append(batch_entry)
            checkpoint(result)
        result["promoted"].append(promotion_result)
        checkpoint(result)
        print("PROMOTED", json.dumps(promotion_result), flush=True)

    full_states = make_valid_states(
        FULL_BATCH, architecture.STATE_LEN, architecture.NUM_CLASSES
    )
    original = jax.jit(lambda x: original_apply(params, x, dtype=jnp.bfloat16))
    original_first, original_steady, original_samples = measure(
        lambda: original(full_states), warmups=3, repeats=9
    )
    oracle = original(full_states)
    result["full_model"].append({
        "id": "original_jax",
        "status": "valid",
        "compile_and_first_seconds": original_first,
        "steady_seconds_median": original_steady,
        "states_per_second": FULL_BATCH / original_steady,
        "samples": original_samples,
    })
    checkpoint(result)

    top_configs = [
        {key: entry[key] for key in (
            "id", "fusion", "bm", "bk", "bn", "fp32_statistics"
        )}
        for entry in promoted[:2]
    ]
    if top_configs:
        best = top_configs[0]
        for fusion in ("separate", "per_layer", "per_block"):
            candidate_config = {**best, "id": f"full-{fusion}", "fusion": fusion}
            try:
                call = jax.jit(lambda x, c=candidate_config: stream1_layernorm_pallas_inference(
                    x, weights, architecture,
                    input_encoding=InputEncodingKind.EMBEDDING_GATHER,
                    bm=c["bm"], bk_input=c["bk"], bn_input=c["bn"],
                    bk_hidden=c["bk"], bn_hidden=c["bn"],
                    bk_output=256, bn_output=128,
                    layernorm_fusion=c["fusion"],
                    fp32_statistics=c["fp32_statistics"],
                ))
                first, steady, samples = measure(
                    lambda: call(full_states), warmups=3, repeats=9
                )
                check = correctness(call(full_states), oracle, output_head=True)
                entry = {
                    **candidate_config,
                    "status": "valid" if check["valid"] else "correctness_failed",
                    "compile_and_first_seconds": first,
                    "steady_seconds_median": steady,
                    "states_per_second": FULL_BATCH / steady,
                    "speedup_vs_original": original_steady / steady,
                    "correctness": check,
                    "samples": samples,
                }
            except Exception as error:
                entry = {
                    **candidate_config,
                    "status": "rejected_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            result["full_model"].append(entry)
            checkpoint(result)
            print("FULL", json.dumps(entry), flush=True)

    valid_full = select_fastest_valid(
        [entry for entry in result["full_model"] if entry["id"] != "original_jax"],
        1,
    )
    if valid_full:
        result["scaling"] = {
            "status": "deferred_to_dedicated_1_8_tpu_scan",
            "winner": valid_full[0]["id"],
            "reason": "full-model correctness passed; run sharded scaling with winner only",
        }
    else:
        result["scaling"] = {
            "status": "skipped_no_correct_pallas_full_model",
            "reason": "no Pallas full-model candidate passed the 99% argmax gate",
        }
    checkpoint(result)
    print("RESULT_PATH", RESULT_PATH, flush=True)


if __name__ == "__main__":
    main()
