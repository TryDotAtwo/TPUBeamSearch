from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


STATE_SIZE = 150
NUM_CLASSES = 150
INTERNAL_BATCH = 16_384
ACTUAL_LOCAL_BEAM = 2_097_152
STATE_TILE_ROWS = 4096


def valid_state_tile(rows: int = STATE_TILE_ROWS) -> np.ndarray:
    row = np.arange(rows, dtype=np.uint32)[:, None]
    column = np.arange(STATE_SIZE, dtype=np.uint32)[None, :]
    mixed = row * np.uint32(0x9E3779B9) + column * np.uint32(0x85EBCA6B)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    return (mixed % NUM_CLASSES).astype(np.uint8)


def find_dataset() -> Path:
    for path in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("artgor/cube555-tpu-artifacts is not attached")


def measure(call, *, warmups: int, repeats: int):
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


def sharded_states(global_batch: int, sharding, tile: np.ndarray):
    def callback(index):
        indices = np.arange(index[0].start, index[0].stop, dtype=np.int64)
        return np.take(tile, indices % tile.shape[0], axis=0)

    return jax.make_array_from_callback(
        (global_batch, STATE_SIZE), sharding, callback
    )


def main():
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import apply as model_apply, load_params_from_pt, num_params

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    tile = valid_state_tile()
    devices = jax.devices()
    if len(devices) < 8:
        raise RuntimeError(f"need 8 TPU devices, found {len(devices)}")

    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in devices],
        },
        "contract": {
            "selected_implementation": "original_jax_model_apply",
            "selection_reason": "Pallas failed representative argmax correctness and was slower",
            "checkpoint": "q555_2k_BEST.pt",
            "parameters": num_params(params),
            "state_size": STATE_SIZE,
            "num_classes": NUM_CLASSES,
            "state_tile_rows": STATE_TILE_ROWS,
            "internal_batch": INTERNAL_BATCH,
            "actual_local_beam": ACTUAL_LOCAL_BEAM,
            "chunks_per_device": ACTUAL_LOCAL_BEAM // INTERNAL_BATCH,
            "dtype": "bfloat16",
        },
    }

    single_states = jnp.asarray(tile[np.arange(INTERNAL_BATCH) % len(tile)])
    single = jax.jit(lambda x: model_apply(params, x, dtype=jnp.bfloat16))
    first, steady, samples = measure(
        lambda: single(single_states), warmups=5, repeats=21
    )
    single_output = single(single_states)
    if single_output.shape != (INTERNAL_BATCH, 30) or not bool(
        jnp.all(jnp.isfinite(single_output))
    ):
        raise AssertionError("invalid single-device output")
    result["single_device"] = {
        "local_batch": INTERNAL_BATCH,
        "compile_and_first_seconds": first,
        "steady_seconds_median": steady,
        "states_per_second": INTERNAL_BATCH / steady,
        "samples": samples,
    }

    mesh = Mesh(np.asarray(devices[:8]), ("core",))
    sharding = NamedSharding(mesh, P("core", None))
    chunk_states = sharded_states(INTERNAL_BATCH * 8, sharding, tile)
    mapped_chunk = jax.jit(
        jax.shard_map(
            lambda x: model_apply(params, x, dtype=jnp.bfloat16),
            mesh=mesh,
            in_specs=P("core", None),
            out_specs=P("core", None),
            check_vma=False,
        )
    )
    first, steady, samples = measure(
        lambda: mapped_chunk(chunk_states), warmups=5, repeats=21
    )
    result["eight_device_chunk"] = {
        "local_batch": INTERNAL_BATCH,
        "global_batch": INTERNAL_BATCH * 8,
        "compile_and_first_seconds": first,
        "steady_seconds_median": steady,
        "states_per_second": INTERNAL_BATCH * 8 / steady,
        "parallel_efficiency": (INTERNAL_BATCH * 8 / steady)
        / (8 * result["single_device"]["states_per_second"]),
        "samples": samples,
    }

    scan_states = sharded_states(ACTUAL_LOCAL_BEAM * 8, sharding, tile)

    def local_scan(states):
        chunks = states.reshape(
            ACTUAL_LOCAL_BEAM // INTERNAL_BATCH, INTERNAL_BATCH, STATE_SIZE
        )

        def body(_, chunk):
            return None, model_apply(params, chunk, dtype=jnp.bfloat16)

        _, outputs = jax.lax.scan(body, None, chunks)
        return outputs.reshape(ACTUAL_LOCAL_BEAM, 30)

    mapped_scan = jax.jit(
        jax.shard_map(
            local_scan,
            mesh=mesh,
            in_specs=P("core", None),
            out_specs=P("core", None),
            check_vma=False,
        )
    )
    first, steady, samples = measure(
        lambda: mapped_scan(scan_states), warmups=2, repeats=7
    )
    global_beam = ACTUAL_LOCAL_BEAM * 8
    result["eight_device_scan"] = {
        "local_batch": ACTUAL_LOCAL_BEAM,
        "global_batch": global_beam,
        "chunks_per_device": ACTUAL_LOCAL_BEAM // INTERNAL_BATCH,
        "compile_and_first_seconds": first,
        "steady_seconds_median": steady,
        "states_per_second": global_beam / steady,
        "samples": samples,
    }

    path = Path("/kaggle/working/stream1_layernorm_scaling.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(result), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
