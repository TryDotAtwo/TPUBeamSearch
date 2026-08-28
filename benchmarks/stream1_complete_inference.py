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


def measure(call, *, warmups=10, repeats=31):
    started = time.perf_counter()
    output = call()
    output.block_until_ready()
    compile_and_first = time.perf_counter() - started
    for _ in range(warmups):
        call().block_until_ready()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call().block_until_ready()
        samples.append(time.perf_counter() - started)
    return output, compile_and_first, statistics.median(samples), samples


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
    reference = make_jitted_stream1_inference(architecture, backend="reference")
    pallas = make_jitted_stream1_inference(architecture, backend="pallas")

    reference_output = reference(states, weights)
    reference_output.block_until_ready()
    output, compile_and_first, steady, samples = measure(
        lambda: pallas(states, weights)
    )
    max_error = float(
        jnp.max(
            jnp.abs(
                output.astype(jnp.float32)
                - reference_output.astype(jnp.float32)
            )
        )
    )
    absolute_error = jnp.abs(
        output.astype(jnp.float32) - reference_output.astype(jnp.float32)
    )
    mean_error = float(jnp.mean(absolute_error))
    argmax_agreement = float(
        jnp.mean(
            (
                jnp.argmax(output, axis=1)
                == jnp.argmax(reference_output, axis=1)
            ).astype(jnp.float32)
        )
    )

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "architecture": architecture.__dict__,
        "weights": {
            "input": list(weights.input.weight.shape),
            "hidden": list(weights.hidden.weight.shape),
            "residual_blocks": len(weights.residuals),
            "residual_matrices": [
                [list(block.first.weight.shape), list(block.second.weight.shape)]
                for block in weights.residuals
            ],
            "output": list(weights.output.weight.shape),
        },
        "jit_contract": {
            "static": ["architecture", "batch shape", "dtypes", "Pallas tiles"],
            "dynamic": ["states", "folded weights", "folded biases"],
        },
        "run": {
            "batch": batch,
            "checkpoint": str(checkpoint_path),
            "inspected_checkpoints": inspected,
            "compile_and_first_seconds": compile_and_first,
            "steady_seconds_median": steady,
            "steady_samples": samples,
            "max_reference_error": max_error,
            "mean_reference_error": mean_error,
            "argmax_agreement": argmax_agreement,
            "checksum": float(jnp.sum(output.astype(jnp.float32))),
        },
    }
    path = Path("/kaggle/working/stream1_complete_inference.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_JSON", json.dumps(result), flush=True)
    print("RESULT_PATH", path, flush=True)
    if not np.isfinite(max_error):
        raise AssertionError(f"Pallas/reference error is not finite: {max_error}")
    if max_error > 0.25:
        raise AssertionError(f"Pallas/reference max error exceeds 0.25: {max_error}")


if __name__ == "__main__":
    main()
