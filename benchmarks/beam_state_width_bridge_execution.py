"""Eight-TPU correctness gate for compact 150/30 final materialization.

No timing is reported: compilation/execution here establish only physical
feasibility and exact bytes for one bounded final-request fixture per core.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.beam_final_materialize import pallas_materialize_final
from tpu_beam_search.beam_final_scatter import pallas_scatter_compact_final_responses
from tpu_beam_search.beam_state_width_bridge import (
    pallas_compact_state_rows, pallas_expand_generator_rows,
    pallas_expand_state_rows,
)


def make_inputs():
    parents = np.zeros((8, 128, 160), np.uint8)
    generators = np.zeros((8, 30, 160), np.int32)
    requests = np.zeros((8, 4, 128), np.uint32)
    counts = np.ones((8, 1), np.uint32)
    target_counts = np.full((8, 1), 128, np.uint32)
    frontier = np.zeros((8, 128, 160), np.uint8)
    for device in range(8):
        frontier[device, :, :150] = np.uint8(31 + device)
        for parent in range(128):
            parents[device, parent, :150] = (
                np.arange(150, dtype=np.uint32) + device * 37 + parent * 11
            ).astype(np.uint8)
        parents[device, :, 150:160] = 0xa5  # Dirty nonlogical source tail.
        for move in range(30):
            generators[device, move, :150] = (
                np.arange(149, -1, -1, dtype=np.int32) + move
            ) % 150
        generators[device, :, 150:] = -1  # Must never affect valid output.
        requests[device, 0, 0] = device + 1
        requests[device, 2, 0] = device + 9
        requests[device, 3, 0] = np.uint32(device << 16)
    return parents, generators, requests, counts, target_counts, frontier


def expected(parents, generators, requests, counts, target_counts, frontier):
    wire = np.zeros((8, 128, 160), np.uint8)
    errors = np.zeros((8, 2, 128), np.uint32)
    published = frontier.copy()
    scatter_errors = np.zeros((8, 2, 128), np.uint32)
    for device in range(8):
        if int(counts[device, 0]) != 1:
            raise ValueError("oracle fixture expects one request per device")
        parent = int(requests[device, 0, 0])
        target = int(requests[device, 2, 0])
        move = int((requests[device, 3, 0] >> np.uint32(16)) & np.uint32(255))
        if not (0 <= parent < 128 and 0 <= target < int(target_counts[device, 0])
                and 0 <= move < 30):
            raise ValueError("oracle fixture has invalid request")
        for position in range(150):
            selected = int(generators[device, move, position])
            wire[device, 0, position] = parents[device, parent, selected]
        for byte in range(4):
            wire[device, 0, 150 + byte] = (target >> (8 * byte)) & 255
        published[device, target, :150] = wire[device, 0, :150]
        published[device, target, 150:] = 0
        errors[device, 1, 0] = np.uint32(0xffffffff)
        scatter_errors[device, 1, 0] = np.uint32(0xffffffff)
    return wire, published, errors, scatter_errors


def _hash(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def validate(wire, published, errors, scatter_errors,
             expected_wire, expected_published, expected_errors, expected_scatter_errors):
    wire, published, errors, scatter_errors = map(
        np.asarray, (wire, published, errors, scatter_errors),
    )
    shape_ok = (wire.shape == expected_wire.shape
                and published.shape == expected_published.shape
                and errors.shape == expected_errors.shape
                and scatter_errors.shape == expected_scatter_errors.shape)
    dtype_ok = (wire.dtype == np.uint8 and published.dtype == np.uint8
                and errors.dtype == np.uint32 and scatter_errors.dtype == np.uint32)
    exact_by_device = [
        bool(np.array_equal(wire[device], expected_wire[device])
             and np.array_equal(published[device], expected_published[device])
             and np.array_equal(errors[device], expected_errors[device])
             and np.array_equal(scatter_errors[device], expected_scatter_errors[device]))
        for device in range(8)
    ] if shape_ok else [False] * 8
    hash_ok = _hash(wire, published, errors, scatter_errors) == _hash(
        expected_wire, expected_published, expected_errors, expected_scatter_errors,
    )
    return {
        "exact": bool(shape_ok and dtype_ok and hash_ok and all(exact_by_device)),
        "exact_by_device": exact_by_device, "shape_ok": shape_ok,
        "dtype_ok": dtype_ok, "hash_ok": hash_ok,
        "wire_shape": list(wire.shape), "frontier_shape": list(published.shape),
        "errors_shape": list(errors.shape), "scatter_errors_shape": list(scatter_errors.shape),
        "wire_dtype": str(wire.dtype), "frontier_dtype": str(published.dtype),
        "errors_dtype": str(errors.dtype), "scatter_errors_dtype": str(scatter_errors.dtype),
        "actual_sha256": _hash(wire, published, errors, scatter_errors),
    }


def make_sharded_call(mesh, *, interpret=False):
    partition = jax.sharding.PartitionSpec("core")

    def local(parents, generators, requests, counts, target_counts, frontier):
        expanded_parents = pallas_expand_state_rows(
            parents[0], state_len=150, kernel_width=256, interpret=interpret,
        )
        expanded_generators = pallas_expand_generator_rows(
            generators[0], state_len=150, kernel_width=256, interpret=interpret,
        )
        wire, errors = pallas_materialize_final(
            expanded_parents, expanded_generators, requests[0], counts[0],
            target_counts[0], state_len=150, interpret=interpret,
        )
        compact = pallas_compact_state_rows(
            wire, state_len=150, storage_width=160, interpret=interpret,
        )
        published, scatter_errors = pallas_scatter_compact_final_responses(
            frontier[0], wire, counts[0], state_len=150,
            prior_error=errors[:1], interpret=interpret,
        )
        return compact[None], published[None], errors[None], scatter_errors[None]

    return jax.jit(jax.shard_map(
        local, mesh=mesh, in_specs=(partition,) * 6,
        out_specs=(partition,) * 4, check_vma=False,
    ))


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise RuntimeError("requires eight physical TPU devices")
    inputs = make_inputs()
    want_wire, want_frontier, want_errors, want_scatter_errors = expected(*inputs)
    report = {
        "status": "lowering", "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "runtime": {name: importlib.metadata.version(name)
                    for name in ("jax", "jaxlib", "libtpu")},
        "devices": [{"id": device.id, "kind": device.device_kind}
                    for device in devices],
        "state_len": 150, "move_count": 30, "storage_width": 160,
        "kernel_width": 256, "input_sha256": _hash(*inputs),
        "expected_sha256": _hash(want_wire, want_frontier, want_errors, want_scatter_errors),
        "scope": "8-device compact-to-tile/materialize/direct-compact-scatter correctness",
    }
    result_path = output / "state_width_bridge.json"

    def save():
        result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    mesh = jax.sharding.Mesh(np.asarray(devices), ("core",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("core"))
    placed = tuple(jax.device_put(value, sharding) for value in inputs)
    try:
        lowered = make_sharded_call(mesh).lower(*placed)
        (output / "lowered.mlir").write_text(lowered.as_text(), encoding="utf-8")
        report["status"] = "compiling"
        save()
        executable = lowered.compile()
        (output / "compiled.hlo.txt").write_text(executable.as_text(), encoding="utf-8")
        report["status"] = "executing"
        save()
        actual = jax.block_until_ready(executable(*placed))
        actual_wire, actual_frontier, actual_errors, actual_scatter_errors = map(np.asarray, actual)
        report.update(validate(actual_wire, actual_frontier, actual_errors, actual_scatter_errors,
                               want_wire, want_frontier, want_errors, want_scatter_errors))
        report["status"] = "complete" if report["exact"] else "mismatch"
        save()
        if not report["exact"]:
            raise AssertionError(report)
    except Exception as exc:
        report["failure_type"] = type(exc).__name__
        report["failure_message"] = str(exc)
        if report["status"] != "mismatch":
            report["status"] = "error"
        save()
        raise
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
