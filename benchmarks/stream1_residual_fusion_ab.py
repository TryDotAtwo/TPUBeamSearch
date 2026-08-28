from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from stream1_first_layer_ab import (
    find_generator_file,
    find_move_count_checkpoint,
    make_reachable_states,
)
from tpu_beam_search.config import BeamConfig, load_generators
from tpu_beam_search.stream1_inference import (
    Stream1Architecture,
    make_jitted_stream1_inference,
    stream1_weights_from_pytorch_state_dict,
)


VARIANTS = ("separate", "per_block", "pairs")


def main():
    generator_path = find_generator_file()
    beam = BeamConfig.from_generators(generator_path)
    generators = np.asarray(load_generators(generator_path))
    checkpoint_path, state_dict, inspected = find_move_count_checkpoint(
        beam.MOVE_COUNT
    )
    architecture = Stream1Architecture.from_pytorch_state_dict(
        state_dict,
        STATE_LEN=beam.STATE_LEN,
        STATE_STORAGE_LEN=beam.STATE_STORAGE_LEN,
        NUM_CLASSES=beam.NUM_CLASSES,
    )
    weights = stream1_weights_from_pytorch_state_dict(state_dict, architecture)
    del state_dict

    batch = 256
    states = make_reachable_states(
        generators, batch, beam.STATE_LEN, beam.STATE_STORAGE_LEN
    )
    calls = {
        variant: make_jitted_stream1_inference(
            architecture,
            backend="pallas",
            residual_fusion=variant,
            bm=256,
            bk_input=128,
            bn_input=512,
            bk_hidden=256,
            bn_hidden=512,
            bk_residual=256,
            bn_residual=512,
            bk_output=512,
            bn_output=256,
        )
        for variant in VARIANTS
    }
    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "architecture": architecture.__dict__,
        "contract": {
            "batch": batch,
            "checkpoint": str(checkpoint_path),
            "inspected_checkpoints": inspected,
            "warmups": 10,
            "samples": 31,
            "tiles": {
                "bm": 256,
                "bk_input": 128,
                "bn_input": 512,
                "bk_hidden": 256,
                "bn_hidden": 512,
                "bk_residual": 256,
                "bn_residual": 512,
                "bk_output": 512,
                "bn_output": 256,
            },
        },
        "variants": {},
        "decision": {},
    }

    outputs = {}
    viable = []
    for variant in VARIANTS:
        started = time.perf_counter()
        try:
            output = calls[variant](states, weights)
            output.block_until_ready()
        except Exception as error:
            result["variants"][variant] = {
                "rejected": type(error).__name__,
                "message": str(error),
            }
            print("REJECTED", variant, repr(error), flush=True)
            continue
        compile_seconds = time.perf_counter() - started
        outputs[variant] = output
        viable.append(variant)
        result["variants"][variant] = {
            "compile_and_first_seconds": compile_seconds,
            "samples": [],
            "checksum": float(jnp.sum(output.astype(jnp.float32))),
        }

    if "separate" not in viable:
        raise RuntimeError("baseline did not compile")
    baseline = outputs["separate"]
    for variant in viable:
        error = float(
            jnp.max(
                jnp.abs(
                    outputs[variant].astype(jnp.float32)
                    - baseline.astype(jnp.float32)
                )
            )
        )
        result["variants"][variant]["max_baseline_error"] = error
        if error != 0.0:
            raise AssertionError(f"{variant} differs from separate: {error}")

    for _ in range(10):
        for variant in viable:
            calls[variant](states, weights).block_until_ready()

    for repeat in range(31):
        offset = repeat % len(viable)
        order = viable[offset:] + viable[:offset]
        for variant in order:
            started = time.perf_counter()
            calls[variant](states, weights).block_until_ready()
            result["variants"][variant]["samples"].append(
                time.perf_counter() - started
            )

    baseline_median = None
    for variant in viable:
        entry = result["variants"][variant]
        entry["steady_seconds_median"] = statistics.median(entry["samples"])
        if variant == "separate":
            baseline_median = entry["steady_seconds_median"]
    for variant in viable:
        entry = result["variants"][variant]
        entry["speedup_vs_separate"] = baseline_median / entry[
            "steady_seconds_median"
        ]
    best = min(
        viable,
        key=lambda variant: result["variants"][variant]["steady_seconds_median"],
    )
    result["decision"] = {
        "best_variant": best,
        "best_seconds": result["variants"][best]["steady_seconds_median"],
        "best_speedup_vs_separate": result["variants"][best][
            "speedup_vs_separate"
        ],
    }

    path = Path("/kaggle/working/stream1_residual_fusion_ab.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_JSON", json.dumps(result), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
