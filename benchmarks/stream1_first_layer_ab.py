from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from tpu_beam_search.config import BeamConfig, load_generators
from tpu_beam_search.stream1_pallas import (
    pallas_embedding_sum_linear,
    pallas_folded_input_linear,
)
from tpu_beam_search.stream1_reference import folded_input_linear


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    return {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in checkpoint.items()
    }


def fold_linear_bn(state_dict, linear: str, bn: str):
    weight = state_dict[f"{linear}.weight"].detach().float()
    bias = state_dict.get(f"{linear}.bias")
    bias = torch.zeros(weight.shape[0]) if bias is None else bias.detach().float()
    gamma = state_dict[f"{bn}.weight"].detach().float()
    beta = state_dict[f"{bn}.bias"].detach().float()
    mean = state_dict[f"{bn}.running_mean"].detach().float()
    variance = state_dict[f"{bn}.running_var"].detach().float()
    scale = gamma / torch.sqrt(variance + 1e-5)
    return weight * scale[:, None], (bias - mean) * scale + beta


def find_generator_file() -> Path:
    matches = list(Path("/kaggle/input").rglob("p900.json"))
    if not matches:
        raise FileNotFoundError("p900.json is not attached")
    return matches[0]


def find_move_count_checkpoint(MOVE_COUNT: int):
    inspected = []
    preferred = sorted(Path("/kaggle/input").rglob("*.pth"), key=lambda p: "p900" not in p.name)
    for path in preferred:
        state_dict = unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        output = state_dict.get("output_layer.weight")
        inspected.append({"path": str(path), "output_shape": list(output.shape) if output is not None else None})
        if output is not None and int(output.shape[0]) == MOVE_COUNT and "bn1.weight" in state_dict:
            return path, state_dict, inspected
        del state_dict
    raise RuntimeError(f"no batchnorm-folded MOVE_COUNT head checkpoint: {inspected}")


def make_reachable_states(generators: np.ndarray, rows: int, STATE_LEN: int, STATE_STORAGE_LEN: int):
    rng = np.random.default_rng(20260828)
    states = np.zeros((rows, STATE_STORAGE_LEN), dtype=np.uint8)
    for row in range(rows):
        state = np.arange(STATE_LEN, dtype=np.uint8)
        for move in rng.integers(0, generators.shape[0], size=16):
            state = state[generators[move, :STATE_LEN]]
        states[row, :STATE_LEN] = state
    return jnp.asarray(states)


def timed(call, repeats=7):
    first_started = time.perf_counter()
    output = call()
    output.block_until_ready()
    first_seconds = time.perf_counter() - first_started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = call()
        output.block_until_ready()
        samples.append(time.perf_counter() - started)
    return output, first_seconds, statistics.median(samples), samples


def main():
    generator_path = find_generator_file()
    config = BeamConfig.from_generators(generator_path)
    generators = np.asarray(load_generators(generator_path))
    checkpoint_path, state_dict, inspected = find_move_count_checkpoint(config.MOVE_COUNT)
    weight_out_in, bias = fold_linear_bn(state_dict, "input_layer", "bn1")
    weight = jnp.asarray(weight_out_in.t().numpy(), dtype=jnp.bfloat16)
    bias = jnp.asarray(bias.numpy(), dtype=jnp.bfloat16)
    del state_dict, weight_out_in

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "MOVE_COUNT": config.MOVE_COUNT,
            "STATE_LEN": config.STATE_LEN,
            "STATE_STORAGE_LEN": config.STATE_STORAGE_LEN,
            "NUM_CLASSES": config.NUM_CLASSES,
            "input_weight_shape": list(weight.shape),
            "checkpoint": str(checkpoint_path),
            "generator_path": str(generator_path),
            "inspected_checkpoints": inspected,
        },
        "batches": {},
    }

    for batch in (32, 128, 256):
        states = make_reachable_states(
            generators, batch, config.STATE_LEN, config.STATE_STORAGE_LEN
        )

        one_hot_call = jax.jit(
            lambda s, w, b: pallas_folded_input_linear(
                s,
                w,
                b,
                STATE_LEN=config.STATE_LEN,
                NUM_CLASSES=config.NUM_CLASSES,
                bm=128,
                bk=128,
                bn=256,
            )
        )

        def embedding_chunked(s, w, b):
            chunks = []
            for begin in range(0, s.shape[0], 32):
                chunks.append(
                    pallas_embedding_sum_linear(
                        s[begin : begin + 32],
                        w,
                        b,
                        STATE_LEN=config.STATE_LEN,
                        NUM_CLASSES=config.NUM_CLASSES,
                        bn=128,
                    )
                )
            return jnp.concatenate(chunks, axis=0)

        embedding_call = jax.jit(embedding_chunked)
        one_hot_out, one_hot_first, one_hot_steady, one_hot_samples = timed(
            lambda: one_hot_call(states, weight, bias)
        )
        embedding_out, embedding_first, embedding_steady, embedding_samples = timed(
            lambda: embedding_call(states.astype(jnp.uint32), weight, bias)
        )

        max_cross_error = float(jnp.max(jnp.abs(one_hot_out - embedding_out)))
        batch_result = {
            "virtual_one_hot_mxu": {
                "compile_and_first_seconds": one_hot_first,
                "steady_seconds_median": one_hot_steady,
                "steady_samples": one_hot_samples,
            },
            "embedding_sum_vpu": {
                "compile_and_first_seconds": embedding_first,
                "steady_seconds_median": embedding_steady,
                "steady_samples": embedding_samples,
            },
            "max_cross_error": max_cross_error,
        }
        if batch == 32:
            reference = folded_input_linear(
                states[:, : config.STATE_LEN],
                weight,
                bias,
                NUM_CLASSES=config.NUM_CLASSES,
            )
            batch_result["max_one_hot_reference_error"] = float(
                jnp.max(jnp.abs(one_hot_out - reference))
            )
            batch_result["max_embedding_reference_error"] = float(
                jnp.max(jnp.abs(embedding_out - reference))
            )
        result["batches"][str(batch)] = batch_result
        print("BATCH", batch, json.dumps(batch_result), flush=True)

    result_path = Path("/kaggle/working/stream1_first_layer_ab.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT", result_path, flush=True)


if __name__ == "__main__":
    main()
