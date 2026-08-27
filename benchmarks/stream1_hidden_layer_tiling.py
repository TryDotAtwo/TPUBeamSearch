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
)


def main():
    generator_path = find_generator_file()
    config = BeamConfig.from_generators(generator_path)
    generators = np.asarray(load_generators(generator_path))
    checkpoint_path, state_dict, inspected = find_move_count_checkpoint(config.MOVE_COUNT)
    input_weight_out_in, input_bias = fold_linear_bn(state_dict, "input_layer", "bn1")
    hidden_weight_out_in, hidden_bias = fold_linear_bn(state_dict, "hidden_layer", "bn2")
    input_weight = jnp.asarray(input_weight_out_in.t().numpy(), dtype=jnp.bfloat16)
    input_bias = jnp.asarray(input_bias.numpy(), dtype=jnp.bfloat16)
    hidden_weight = jnp.asarray(hidden_weight_out_in.t().numpy(), dtype=jnp.bfloat16)
    hidden_bias = jnp.asarray(hidden_bias.numpy(), dtype=jnp.bfloat16)
    del state_dict, input_weight_out_in, hidden_weight_out_in

    batch = 256
    states = make_reachable_states(
        generators, batch, config.STATE_LEN, config.STATE_STORAGE_LEN
    )
    first_layer = jax.jit(
        lambda s, w, b: pallas_folded_input_linear(
            s,
            w,
            b,
            STATE_LEN=config.STATE_LEN,
            NUM_CLASSES=config.NUM_CLASSES,
            bm=128,
            bk=128,
            bn=256,
            relu=True,
        )
    )
    hidden1, first_compile, first_steady, first_samples = timed(
        lambda: first_layer(states, input_weight, input_bias)
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
        "first_layer": {
            "compile_and_first_seconds": first_compile,
            "steady_seconds_median": first_steady,
            "steady_samples": first_samples,
            "output_dtype": str(hidden1.dtype),
        },
        "tiling": {},
    }

    best = None
    for bm in (128, 256):
        for bk in (128, 256, 512):
            for bn in (256, 512):
                key = f"bm{bm}_bk{bk}_bn{bn}"
                dense = jax.jit(
                    lambda x, w, b, bm=bm, bk=bk, bn=bn: pallas_dense_linear(
                        x, w, b, bm=bm, bk=bk, bn=bn, relu=True
                    )
                )
                output, compile_seconds, steady_seconds, samples = timed(
                    lambda dense=dense: dense(hidden1, hidden_weight, hidden_bias)
                )
                entry = {
                    "compile_and_first_seconds": compile_seconds,
                    "steady_seconds_median": steady_seconds,
                    "steady_samples": samples,
                    "output_checksum": float(jnp.sum(output.astype(jnp.float32))),
                }
                result["tiling"][key] = entry
                print("TILE", key, json.dumps(entry), flush=True)
                if best is None or steady_seconds < best[1]:
                    best = (key, steady_seconds, bm, bk, bn, output)

    _, best_seconds, best_bm, best_bk, best_bn, best_output = best
    reference = jnp.maximum(
        hidden1.astype(jnp.float32) @ hidden_weight.astype(jnp.float32)
        + hidden_bias.astype(jnp.float32),
        0.0,
    ).astype(jnp.bfloat16)

    pipeline = jax.jit(
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
            bm=best_bm,
            bk=best_bk,
            bn=best_bn,
            relu=True,
        )
    )
    pipeline_output, pipeline_compile, pipeline_steady, pipeline_samples = timed(
        lambda: pipeline(
            states, input_weight, input_bias, hidden_weight, hidden_bias
        )
    )

    hidden1_bytes = batch * input_weight.shape[1] * 2
    hbm_roundtrip_bytes = hidden1_bytes * 2
    hbm_bandwidth_bytes_per_second = 412.5e9
    peak_bandwidth_roundtrip_seconds = (
        hbm_roundtrip_bytes / hbm_bandwidth_bytes_per_second
    )
    result["decision"] = {
        "best_tile": f"bm{best_bm}_bk{best_bk}_bn{best_bn}",
        "best_hidden_seconds": best_seconds,
        "max_best_reference_error": float(
            jnp.max(
                jnp.abs(
                    best_output.astype(jnp.float32) - reference.astype(jnp.float32)
                )
            )
        ),
        "pipeline_compile_and_first_seconds": pipeline_compile,
        "pipeline_steady_seconds_median": pipeline_steady,
        "pipeline_steady_samples": pipeline_samples,
        "pipeline_checksum": float(jnp.sum(pipeline_output.astype(jnp.float32))),
        "hidden1_hbm_roundtrip_bytes": int(hbm_roundtrip_bytes),
        "peak_hbm_bandwidth_bytes_per_second": hbm_bandwidth_bytes_per_second,
        "peak_bandwidth_roundtrip_seconds_estimate": peak_bandwidth_roundtrip_seconds,
        "peak_bandwidth_roundtrip_fraction_of_pipeline": (
            peak_bandwidth_roundtrip_seconds / pipeline_steady
        ),
    }

    result_path = Path("/kaggle/working/stream1_hidden_layer_tiling.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DECISION", json.dumps(result["decision"]), flush=True)
    print("RESULT", result_path, flush=True)


if __name__ == "__main__":
    main()
