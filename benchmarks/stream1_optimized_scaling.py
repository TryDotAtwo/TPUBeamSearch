from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.stream1_first_layer_ab import (
    find_generator_file,
    find_move_count_checkpoint,
    make_reachable_states,
)
from tpu_beam_search.config import BeamConfig, load_generators
from tpu_beam_search.sharding import make_sharded_inference
from tpu_beam_search.stream1_inference import (
    Stream1Architecture,
    make_jitted_stream1_inference,
    stream1_pallas_inference,
    stream1_weights_from_pytorch_state_dict,
)


LOCAL_BATCH = 32768
CORE_COUNTS = (1, 2, 4, 8)
WARMUPS = 5
REPEATS = 21
OPTIMIZED_OPTIONS = {
    "residual_fusion": "separate",
    "bm": 1024,
    "bk_input": 128,
    "bn_input": 1536,
    "bk_hidden": 256,
    "bn_hidden": 512,
    "prefix_pipeline_buffer_count": 0,
    "prefix_pipeline_lookahead": False,
    "bk_residual": 256,
    "bn_residual": 512,
    "bk_output": 512,
    "bn_output": 256,
}


def measure(call):
    started = time.perf_counter()
    output = call()
    output.block_until_ready()
    first = time.perf_counter() - started
    for _ in range(WARMUPS):
        call().block_until_ready()
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        call().block_until_ready()
        samples.append(time.perf_counter() - started)
    return output, first, statistics.median(samples), samples


def main():
    generator_path = find_generator_file()
    beam = BeamConfig.from_generators(generator_path)
    generators = np.asarray(load_generators(generator_path))
    checkpoint_path, state_dict, inspected = find_move_count_checkpoint(beam.MOVE_COUNT)
    architecture = Stream1Architecture.from_pytorch_state_dict(
        state_dict,
        STATE_LEN=beam.STATE_LEN,
        STATE_STORAGE_LEN=beam.STATE_STORAGE_LEN,
        NUM_CLASSES=beam.NUM_CLASSES,
    )
    weights = stream1_weights_from_pytorch_state_dict(state_dict, architecture)
    del state_dict

    # Fresh correctness gate for the selected production candidate.
    check_states = make_reachable_states(
        generators, 256, beam.STATE_LEN, beam.STATE_STORAGE_LEN
    )
    reference = make_jitted_stream1_inference(architecture, backend="reference")
    candidate = make_jitted_stream1_inference(
        architecture, backend="pallas", **OPTIMIZED_OPTIONS
    )
    reference_output = reference(check_states, weights)
    candidate_output = candidate(check_states, weights)
    reference_output.block_until_ready()
    candidate_output.block_until_ready()
    max_reference_error = float(jnp.max(jnp.abs(
        candidate_output.astype(jnp.float32) - reference_output.astype(jnp.float32)
    )))
    if not np.isfinite(max_reference_error) or max_reference_error > 0.25:
        raise AssertionError(f"optimized inference exceeds BF16 gate: {max_reference_error}")

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": str(checkpoint_path),
            "inspected_checkpoints": inspected,
            "architecture": architecture.__dict__,
            "local_batch": LOCAL_BATCH,
            "core_counts": list(CORE_COUNTS),
            "warmups": WARMUPS,
            "repeats": REPEATS,
            "options": OPTIMIZED_OPTIONS,
            "max_reference_error_batch256": max_reference_error,
        },
        "scaling": {},
    }

    devices = jax.devices()
    single_throughput = None
    for core_count in CORE_COUNTS:
        if core_count > len(devices):
            continue
        mesh = Mesh(np.asarray(devices[:core_count]), ("core",))
        state_sharding = NamedSharding(mesh, P("core", None))
        replicated = NamedSharding(mesh, P())
        global_batch = LOCAL_BATCH * core_count
        states_host = make_reachable_states(
            generators, global_batch, beam.STATE_LEN, beam.STATE_STORAGE_LEN
        )
        states_sharded = jax.device_put(states_host, state_sharding)
        weights_replicated = jax.tree.map(
            lambda value: jax.device_put(value, replicated), weights
        )

        def local_infer(local_states, local_weights):
            return stream1_pallas_inference(
                local_states, local_weights, architecture, **OPTIMIZED_OPTIONS
            )

        mapped = make_sharded_inference(
            local_infer,
            mesh=mesh,
            weights_example=weights,
        )
        output, first, steady, samples = measure(
            lambda: mapped(states_sharded, weights_replicated)
        )
        if not bool(jnp.all(jnp.isfinite(output.astype(jnp.float32)))):
            raise AssertionError(f"{core_count} cores produced non-finite logits")
        throughput = global_batch / steady
        if single_throughput is None:
            single_throughput = throughput
        entry = {
            "cores": core_count,
            "local_batch": LOCAL_BATCH,
            "global_batch": global_batch,
            "compile_and_first_seconds": first,
            "steady_seconds_median": steady,
            "states_per_second": throughput,
            "speedup_vs_one_core": throughput / single_throughput,
            "parallel_efficiency": throughput / single_throughput / core_count,
            "samples": samples,
            "checksum": float(jnp.sum(output.astype(jnp.float32))),
        }
        result["scaling"][f"cores{core_count}"] = entry
        print("SCALING", core_count, json.dumps(entry), flush=True)

    path = Path("/kaggle/working/stream1_optimized_scaling.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_JSON", json.dumps(result), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
