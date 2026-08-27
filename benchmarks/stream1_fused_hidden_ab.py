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
    pallas_folded_input_linear,
    pallas_fused_folded_hidden,
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
    input_weight = jnp.asarray(
        input_weight_out_in.t().numpy(), dtype=jnp.bfloat16
    )
    input_bias = jnp.asarray(input_bias.numpy(), dtype=jnp.bfloat16)
    hidden_weight = jnp.asarray(
        hidden_weight_out_in.t().numpy(), dtype=jnp.bfloat16
    )
    hidden_bias = jnp.asarray(hidden_bias.numpy(), dtype=jnp.bfloat16)
    del state_dict, input_weight_out_in, hidden_weight_out_in

    batch = 256
    states = make_reachable_states(
        generators, batch, config.STATE_LEN, config.STATE_STORAGE_LEN
    )

    separate = jax.jit(
        lambda s, w1, b1, w2, b2: pallas_dense_linear(
            pallas_folded_input_linear(
                s,
                w1,
                b1,
                STATE_LEN=config.STATE_LEN,
                NUM_CLASSES=config.NUM_CLASSES,
                bm=128,
                bk=128,
                bn=256,
                relu=True,
            ),
            w2,
            b2,
            bm=128,
            bk=128,
            bn=512,
            relu=True,
        )
    )
    separate_output, separate_compile, separate_seconds, separate_samples = timed(
        lambda: separate(
            states, input_weight, input_bias, hidden_weight, hidden_bias
        )
    )

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
            "checkpoint": str(checkpoint_path),
            "inspected_checkpoints": inspected,
            "batch": batch,
            "timing_warmups": 10,
            "timing_samples": 31,
        },
        "separate": {
            "tile": {
                "bm": 128,
                "bk_input": 128,
                "bn_input": 256,
                "bk_hidden": 128,
                "bn_hidden": 512,
            },
            "compile_and_first_seconds": separate_compile,
            "steady_seconds_median": separate_seconds,
            "steady_samples": separate_samples,
            "checksum": float(jnp.sum(separate_output.astype(jnp.float32))),
        },
        "fused": {},
    }

    best = None
    for bm in (128, 256):
        for bn_input in (256, 512):
            for bk_hidden in (128, 256):
                key = (
                    f"bm{bm}_bki128_bni{bn_input}_"
                    f"bkh{bk_hidden}_bnh512"
                )
                fused = jax.jit(
                    lambda s, w1, b1, w2, b2,
                    bm=bm, bn_input=bn_input, bk_hidden=bk_hidden:
                    pallas_fused_folded_hidden(
                        s,
                        w1,
                        b1,
                        w2,
                        b2,
                        STATE_LEN=config.STATE_LEN,
                        NUM_CLASSES=config.NUM_CLASSES,
                        bm=bm,
                        bk_input=128,
                        bn_input=bn_input,
                        bk_hidden=bk_hidden,
                        bn_hidden=512,
                    )
                )
                output, compile_seconds, steady_seconds, samples = timed(
                    lambda fused=fused: fused(
                        states,
                        input_weight,
                        input_bias,
                        hidden_weight,
                        hidden_bias,
                    )
                )
                max_error = float(
                    jnp.max(
                        jnp.abs(
                            output.astype(jnp.float32)
                            - separate_output.astype(jnp.float32)
                        )
                    )
                )
                entry = {
                    "compile_and_first_seconds": compile_seconds,
                    "steady_seconds_median": steady_seconds,
                    "steady_samples": samples,
                    "max_separate_error": max_error,
                    "checksum": float(jnp.sum(output.astype(jnp.float32))),
                    "speedup_over_separate": separate_seconds / steady_seconds,
                }
                result["fused"][key] = entry
                print("FUSED", key, json.dumps(entry), flush=True)
                if best is None or steady_seconds < best[1]:
                    best = (key, steady_seconds, max_error)

    best_key, best_seconds, best_error = best
    result["decision"] = {
        "best_fused_tile": best_key,
        "best_fused_seconds": best_seconds,
        "separate_seconds": separate_seconds,
        "speedup": separate_seconds / best_seconds,
        "saved_seconds": separate_seconds - best_seconds,
        "max_separate_error": best_error,
    }
    result_path = Path("/kaggle/working/stream1_fused_hidden_ab.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DECISION", json.dumps(result["decision"]), flush=True)
    print("RESULT", result_path, flush=True)


if __name__ == "__main__":
    main()
