from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from stream1_first_layer_ab import (
    find_generator_file,
    find_move_count_checkpoint,
    fold_linear_bn,
    make_reachable_states,
    timed,
)
from tpu_beam_search.config import BeamConfig, load_generators
from tpu_beam_search.stream1_pallas import (
    pallas_dense_linear,
    pallas_fused_folded_hidden,
    pallas_fused_mlp,
)


def main():
    generator_path = find_generator_file()
    config = BeamConfig.from_generators(generator_path)
    generators = np.asarray(load_generators(generator_path))
    checkpoint_path, state_dict, inspected = find_move_count_checkpoint(
        config.MOVE_COUNT
    )
    input_weight_out_in, input_bias = fold_linear_bn(
        state_dict, "input_layer", "bn1"
    )
    hidden_weight_out_in, hidden_bias = fold_linear_bn(
        state_dict, "hidden_layer", "bn2"
    )
    output_weight_out_in = state_dict["output_layer.weight"]
    output_bias_tensor = state_dict["output_layer.bias"]
    input_weight = jnp.asarray(
        input_weight_out_in.t().numpy(), dtype=jnp.bfloat16
    )
    input_bias = jnp.asarray(input_bias.numpy(), dtype=jnp.bfloat16)
    hidden_weight = jnp.asarray(
        hidden_weight_out_in.t().numpy(), dtype=jnp.bfloat16
    )
    hidden_bias = jnp.asarray(hidden_bias.numpy(), dtype=jnp.bfloat16)
    output_weight = jnp.asarray(
        output_weight_out_in.t().numpy(), dtype=jnp.bfloat16
    )
    output_bias = jnp.asarray(output_bias_tensor.numpy(), dtype=jnp.bfloat16)
    del state_dict, input_weight_out_in, hidden_weight_out_in

    batch = 256
    states = make_reachable_states(
        generators, batch, config.STATE_LEN, config.STATE_STORAGE_LEN
    )
    configurations = [
        (bk_output, bn_output)
        for bk_output in (128, 256, 512)
        for bn_output in (128, 256)
    ]
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
            "input_weight_shape": list(input_weight.shape),
            "hidden_weight_shape": list(hidden_weight.shape),
            "output_weight_shape": list(output_weight.shape),
            "checkpoint": str(checkpoint_path),
            "inspected_checkpoints": inspected,
            "batch": batch,
            "timing_warmups": 10,
            "timing_samples": 31,
            "fixed_prefix_tile": {
                "bm": 256,
                "bk_input": 128,
                "bn_input": 512,
                "bk_hidden": 256,
                "bn_hidden": 512,
            },
        },
        "separate_head": {},
        "fused_head": {},
    }

    separate_outputs = {}
    best_separate = None
    for bk_output, bn_output in configurations:
        key = f"bko{bk_output}_bno{bn_output}"
        separate = jax.jit(
            lambda s, w1, b1, w2, b2, w3, b3,
            bk_output=bk_output, bn_output=bn_output:
            pallas_dense_linear(
                pallas_fused_folded_hidden(
                    s,
                    w1,
                    b1,
                    w2,
                    b2,
                    STATE_LEN=config.STATE_LEN,
                    NUM_CLASSES=config.NUM_CLASSES,
                    bm=256,
                    bk_input=128,
                    bn_input=512,
                    bk_hidden=256,
                    bn_hidden=512,
                ),
                w3,
                b3,
                bm=256,
                bk=bk_output,
                bn=bn_output,
                relu=False,
            )
        )
        try:
            output, compile_seconds, steady_seconds, samples = timed(
                lambda separate=separate: separate(
                    states,
                    input_weight,
                    input_bias,
                    hidden_weight,
                    hidden_bias,
                    output_weight,
                    output_bias,
                )
            )
        except Exception as error:
            result["separate_head"][key] = {
                "rejected": type(error).__name__,
                "message": str(error),
            }
            print("SEPARATE_REJECTED", key, repr(error), flush=True)
            continue
        separate_outputs[key] = output
        entry = {
            "compile_and_first_seconds": compile_seconds,
            "steady_seconds_median": steady_seconds,
            "steady_samples": samples,
            "checksum": float(jnp.sum(output.astype(jnp.float32))),
        }
        result["separate_head"][key] = entry
        print("SEPARATE", key, json.dumps(entry), flush=True)
        if best_separate is None or steady_seconds < best_separate[1]:
            best_separate = (key, steady_seconds)

    best_fused = None
    for bk_output, bn_output in configurations:
        key = f"bko{bk_output}_bno{bn_output}"
        if key not in separate_outputs:
            result["fused_head"][key] = {"skipped": "separate rejected"}
            continue
        fused = jax.jit(
            lambda s, w1, b1, w2, b2, w3, b3,
            bk_output=bk_output, bn_output=bn_output:
            pallas_fused_mlp(
                s,
                w1,
                b1,
                w2,
                b2,
                w3,
                b3,
                STATE_LEN=config.STATE_LEN,
                NUM_CLASSES=config.NUM_CLASSES,
                MOVE_COUNT=config.MOVE_COUNT,
                bm=256,
                bk_input=128,
                bn_input=512,
                bk_hidden=256,
                bn_hidden=512,
                bk_output=bk_output,
                bn_output=bn_output,
            )
        )
        try:
            output, compile_seconds, steady_seconds, samples = timed(
                lambda fused=fused: fused(
                    states,
                    input_weight,
                    input_bias,
                    hidden_weight,
                    hidden_bias,
                    output_weight,
                    output_bias,
                )
            )
        except Exception as error:
            result["fused_head"][key] = {
                "rejected": type(error).__name__,
                "message": str(error),
            }
            print("FUSED_REJECTED", key, repr(error), flush=True)
            continue
        max_error = float(
            jnp.max(
                jnp.abs(
                    output.astype(jnp.float32)
                    - separate_outputs[key].astype(jnp.float32)
                )
            )
        )
        paired_seconds = result["separate_head"][key][
            "steady_seconds_median"
        ]
        entry = {
            "compile_and_first_seconds": compile_seconds,
            "steady_seconds_median": steady_seconds,
            "steady_samples": samples,
            "checksum": float(jnp.sum(output.astype(jnp.float32))),
            "max_paired_error": max_error,
            "paired_separate_seconds": paired_seconds,
            "paired_speedup": paired_seconds / steady_seconds,
        }
        result["fused_head"][key] = entry
        print("FUSED", key, json.dumps(entry), flush=True)
        if best_fused is None or steady_seconds < best_fused[1]:
            best_fused = (key, steady_seconds, max_error)

    if best_separate is None or best_fused is None:
        raise RuntimeError("no valid separate/fused head pair")
    best_separate_key, best_separate_seconds = best_separate
    best_fused_key, best_fused_seconds, best_fused_error = best_fused
    result["decision"] = {
        "best_separate_tile": best_separate_key,
        "best_separate_seconds": best_separate_seconds,
        "best_fused_tile": best_fused_key,
        "best_fused_seconds": best_fused_seconds,
        "best_to_best_speedup": best_separate_seconds / best_fused_seconds,
        "best_to_best_saved_seconds": best_separate_seconds - best_fused_seconds,
        "best_fused_max_paired_error": best_fused_error,
    }
    result_path = Path("/kaggle/working/stream1_full_mlp_ab.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DECISION", json.dumps(result["decision"]), flush=True)
    print("RESULT", result_path, flush=True)


if __name__ == "__main__":
    main()
