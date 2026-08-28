from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
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
    stream1_pallas_inference,
    stream1_weights_from_pytorch_state_dict,
)


BATCHES = (64, 128, 256, 512, 1024, 2048, 4096)
BLOCK_ROWS = (128, 256, 512)
WARMUPS = 5
REPEATS = 21


def measure(call):
    started = time.perf_counter()
    output = call()
    output.block_until_ready()
    compile_and_first = time.perf_counter() - started
    for _ in range(WARMUPS):
        call().block_until_ready()
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        call().block_until_ready()
        samples.append(time.perf_counter() - started)
    return output, compile_and_first, statistics.median(samples), samples


def inference_options(bm):
    return {
        "residual_fusion": "separate",
        "bm": bm,
        "bk_input": 128,
        "bn_input": 512,
        "bk_hidden": 256,
        "bn_hidden": 512,
        "bk_residual": 256,
        "bn_residual": 512,
        "bk_output": 512,
        "bn_output": 256,
    }


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
            "batches": list(BATCHES),
            "block_rows": list(BLOCK_ROWS),
            "warmups": WARMUPS,
            "samples": REPEATS,
            "residual_fusion": "separate",
        },
        "single_core": {},
        "scaling": {},
        "decision": {},
    }

    best = None
    best_states = None
    reference_calls = {}
    for batch in BATCHES:
        states = make_reachable_states(
            generators, batch, beam.STATE_LEN, beam.STATE_STORAGE_LEN
        )
        reference = make_jitted_stream1_inference(
            architecture, backend="reference"
        )
        reference_output = reference(states, weights)
        reference_output.block_until_ready()
        reference_calls[batch] = reference_output
        for bm in BLOCK_ROWS:
            key = f"b{batch}_bm{bm}"
            infer = make_jitted_stream1_inference(
                architecture,
                backend="pallas",
                **inference_options(bm),
            )
            try:
                output, compile_seconds, steady, samples = measure(
                    lambda infer=infer, states=states: infer(states, weights)
                )
            except Exception as error:
                result["single_core"][key] = {
                    "rejected": type(error).__name__,
                    "message": str(error),
                }
                print("REJECTED", key, repr(error), flush=True)
                continue
            max_error = float(
                jnp.max(
                    jnp.abs(
                        output.astype(jnp.float32)
                        - reference_output.astype(jnp.float32)
                    )
                )
            )
            if not np.isfinite(max_error) or max_error > 0.25:
                raise AssertionError(f"{key} exceeds BF16 error gate: {max_error}")
            entry = {
                "batch": batch,
                "bm": bm,
                "padded_batch": ((batch + bm - 1) // bm) * bm,
                "compile_and_first_seconds": compile_seconds,
                "steady_seconds_median": steady,
                "states_per_second": batch / steady,
                "nanoseconds_per_state": steady * 1e9 / batch,
                "max_reference_error": max_error,
                "samples": samples,
            }
            result["single_core"][key] = entry
            print("SINGLE", key, json.dumps(entry), flush=True)
            if best is None or entry["states_per_second"] > best[2]:
                best = (batch, bm, entry["states_per_second"], key)
                best_states = states

    if best is None:
        raise RuntimeError("no single-core configuration compiled")
    local_batch, best_bm, single_throughput, best_key = best
    del best_states, reference_calls

    devices = jax.devices()
    for core_count in (1, 2, 4, 8):
        if core_count > len(devices):
            continue
        mesh = Mesh(np.asarray(devices[:core_count]), ("core",))
        state_sharding = NamedSharding(mesh, P("core", None))
        replicated = NamedSharding(mesh, P())
        global_batch = local_batch * core_count
        states_host = make_reachable_states(
            generators, global_batch, beam.STATE_LEN, beam.STATE_STORAGE_LEN
        )
        states_sharded = jax.device_put(states_host, state_sharding)
        weights_replicated = jax.tree.map(
            lambda value: jax.device_put(value, replicated), weights
        )
        weight_specs = jax.tree.map(lambda _: P(), weights)

        def local_infer(local_states, local_weights):
            return stream1_pallas_inference(
                local_states,
                local_weights,
                architecture,
                **inference_options(best_bm),
            )

        mapped = jax.jit(
            jax.shard_map(
                local_infer,
                mesh=mesh,
                in_specs=(P("core", None), weight_specs),
                out_specs=P("core", None),
            )
        )
        key = f"cores{core_count}"
        try:
            output, compile_seconds, steady, samples = measure(
                lambda: mapped(states_sharded, weights_replicated)
            )
        except Exception as error:
            result["scaling"][key] = {
                "rejected": type(error).__name__,
                "message": str(error),
            }
            print("SCALING_REJECTED", key, repr(error), flush=True)
            continue
        entry = {
            "cores": core_count,
            "local_batch": local_batch,
            "global_batch": global_batch,
            "bm": best_bm,
            "compile_and_first_seconds": compile_seconds,
            "steady_seconds_median": steady,
            "states_per_second": global_batch / steady,
            "speedup_vs_single_sweep": (global_batch / steady) / single_throughput,
            "parallel_efficiency": ((global_batch / steady) / single_throughput) / core_count,
            "samples": samples,
            "checksum": float(jnp.sum(output.astype(jnp.float32))),
        }
        result["scaling"][key] = entry
        print("SCALING", key, json.dumps(entry), flush=True)

    result["decision"] = {
        "best_single_core_key": best_key,
        "local_batch": local_batch,
        "bm": best_bm,
        "single_core_states_per_second": single_throughput,
    }
    path = Path("/kaggle/working/stream1_batch_scaling.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_JSON", json.dumps(result), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
