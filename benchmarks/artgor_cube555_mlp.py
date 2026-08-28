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
INTERNAL_BS = 16384
ACTUAL_LOCAL_BEAM = 2_097_152
WARMUPS = 5
REPEATS = 21


def find_dataset() -> Path:
    for path in (
        Path("/kaggle/input/cube555-tpu-artifacts"),
        Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("artgor/cube555-tpu-artifacts is not attached")


def measure(call, *, warmups=WARMUPS, repeats=REPEATS):
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


def sharded_states(global_batch, sharding):
    base = np.arange(STATE_SIZE, dtype=np.uint8)

    def callback(index):
        rows = index[0].stop - index[0].start
        return np.broadcast_to(base, (rows, STATE_SIZE)).copy()

    return jax.make_array_from_callback(
        (global_batch, STATE_SIZE), sharding, callback
    )


def main():
    dataset = find_dataset()
    sys.path.insert(0, str(dataset))
    from jax_model import apply as model_apply, load_params_from_pt, num_params

    params = load_params_from_pt(dataset / "q555_2k_BEST.pt")
    result = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "contract": {
            "checkpoint": "q555_2k_BEST.pt",
            "parameters": num_params(params),
            "state_size": STATE_SIZE,
            "internal_bs": INTERNAL_BS,
            "actual_local_beam": ACTUAL_LOCAL_BEAM,
            "dtype": "bfloat16",
            "warmups": WARMUPS,
            "repeats": REPEATS,
        },
        "single_device": {},
        "eight_device_chunk": {},
        "eight_device_actual_scan": {},
    }

    for batch in (INTERNAL_BS, 32768):
        states = jnp.broadcast_to(
            jnp.arange(STATE_SIZE, dtype=jnp.uint8), (batch, STATE_SIZE)
        )
        call = jax.jit(lambda x: model_apply(params, x, dtype=jnp.bfloat16))
        first, steady, samples = measure(lambda: call(states))
        output = call(states)
        if output.shape != (batch, 30) or not bool(jnp.all(jnp.isfinite(output))):
            raise AssertionError("invalid single-device Q output")
        entry = {
            "batch": batch,
            "compile_and_first_seconds": first,
            "steady_seconds_median": steady,
            "states_per_second": batch / steady,
            "samples": samples,
        }
        result["single_device"][str(batch)] = entry
        print("SINGLE", batch, json.dumps(entry), flush=True)

    devices = jax.devices()
    if len(devices) < 8:
        raise RuntimeError(f"need 8 TPU devices, found {len(devices)}")
    mesh = Mesh(np.asarray(devices[:8]), ("core",))
    sharding = NamedSharding(mesh, P("core", None))

    chunk_states = sharded_states(INTERNAL_BS * 8, sharding)
    mapped_chunk = jax.jit(jax.shard_map(
        lambda x: model_apply(params, x, dtype=jnp.bfloat16),
        mesh=mesh,
        in_specs=P("core", None),
        out_specs=P("core", None),
        check_vma=False,
    ))
    first, steady, samples = measure(lambda: mapped_chunk(chunk_states))
    entry = {
        "local_batch": INTERNAL_BS,
        "global_batch": INTERNAL_BS * 8,
        "compile_and_first_seconds": first,
        "steady_seconds_median": steady,
        "states_per_second": INTERNAL_BS * 8 / steady,
        "samples": samples,
    }
    result["eight_device_chunk"] = entry
    print("EIGHT_CHUNK", json.dumps(entry), flush=True)

    scan_states = sharded_states(ACTUAL_LOCAL_BEAM * 8, sharding)

    def local_scan(states):
        chunks = states.reshape(
            ACTUAL_LOCAL_BEAM // INTERNAL_BS, INTERNAL_BS, STATE_SIZE
        )

        def body(_, chunk):
            return None, model_apply(params, chunk, dtype=jnp.bfloat16)

        _, outputs = jax.lax.scan(body, None, chunks)
        return outputs.reshape(ACTUAL_LOCAL_BEAM, 30)

    mapped_scan = jax.jit(jax.shard_map(
        local_scan,
        mesh=mesh,
        in_specs=P("core", None),
        out_specs=P("core", None),
        check_vma=False,
    ))
    first, steady, samples = measure(
        lambda: mapped_scan(scan_states), warmups=2, repeats=7
    )
    global_beam = ACTUAL_LOCAL_BEAM * 8
    entry = {
        "local_batch": ACTUAL_LOCAL_BEAM,
        "global_batch": global_beam,
        "chunks_per_device": ACTUAL_LOCAL_BEAM // INTERNAL_BS,
        "compile_and_first_seconds": first,
        "steady_seconds_median": steady,
        "states_per_second": global_beam / steady,
        "samples": samples,
    }
    result["eight_device_actual_scan"] = entry
    print("EIGHT_SCAN", json.dumps(entry), flush=True)

    path = Path("/kaggle/working/artgor_cube555_mlp_benchmark.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
