"""Eight-device physical correctness gate for production final-chunk packing."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk


def make_inputs(*, planes=32, scenario="mixed"):
    payload = np.random.default_rng(1729).integers(
        0, 1 << 32, size=(8, planes, 2048), dtype=np.uint32)
    payload[0, 0, 1] = np.uint32(0xffffffff)
    payload[0, 0, 2] = np.uint32(0)
    payload[0, 0, 127] = np.uint32(0x80000000)
    payload[1, 0, 129] = np.uint32(0x7fffffff)
    payload[6, 0, 1920] = np.uint32(0x01000001)
    intervals = np.zeros((8, 3, 128), dtype=np.uint32)
    chunks = np.zeros((8, 1), dtype=np.uint32)
    prior = np.zeros((8, 1, 128), dtype=np.uint32)

    if scenario == "all_live":
        for device in range(8):
            for peer in range(8):
                intervals[device, 0, peer] = (device * 8 + peer) * 24 + 1
                intervals[device, 1, peer] = 128
        return payload, intervals, chunks, prior
    if scenario != "mixed":
        raise ValueError("unknown production packing fixture")

    intervals[0, 0, :3] = [1, 127, 255]
    intervals[0, 1, :3] = [128, 2, 129]
    intervals[1, 0, 0] = 1
    intervals[1, 1, 0] = 256
    chunks[1, 0] = 1
    intervals[3, 1, 0] = 1
    intervals[3, 0, 4] = 2048
    intervals[3, 1, 4] = 1
    intervals[4, 1, 0] = 1
    intervals[4, 2, 0] = 1
    intervals[5, 1, 0] = 1
    prior[5, 0, 0] = 7
    intervals[6, 0, 0] = 1920
    intervals[6, 1, 0] = 128
    intervals[7, 1, 0] = 128
    chunks[7, 0] = np.iinfo(np.uint32).max
    return payload, intervals, chunks, prior


def expected(payload, intervals, chunks, prior):
    wire = np.zeros((8, 8, payload.shape[1], 128), dtype=np.uint32)
    control = np.zeros((8, 8, 2, 128), dtype=np.uint32)
    for device in range(8):
        starts = intervals[device, 0, :8].astype(np.uint64)
        counts = intervals[device, 1, :8].astype(np.uint64)
        bad = (np.any(counts > 2048) or np.any(starts + counts > 2048)
               or intervals[device, 2, 0] != 0 or prior[device, 0, 0] != 0)
        if bad:
            control[device, :, 1, 0] = 1
            continue
        offset = min(int(chunks[device, 0]), 16) * 128
        for peer in range(8):
            count = int(counts[peer])
            if offset >= count:
                continue
            length = min(128, count - offset)
            begin = int(starts[peer]) + offset
            wire[device, peer, :, :length] = payload[device, :, begin:begin + length]
            control[device, peer, 0, 0] = length
    return wire, control


def _hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def validate_outputs(actual_wire, actual_control, want_wire, want_control):
    actual_wire = np.asarray(actual_wire)
    actual_control = np.asarray(actual_control)
    shape_ok = actual_wire.shape == want_wire.shape and actual_control.shape == want_control.shape
    dtype_ok = actual_wire.dtype == np.uint32 and actual_control.dtype == np.uint32
    hash_ok = _hash(actual_wire, actual_control) == _hash(want_wire, want_control)
    exact_by_device = [bool(np.array_equal(actual_wire[device], want_wire[device])
                            and np.array_equal(actual_control[device], want_control[device]))
                       for device in range(8)] if shape_ok else [False] * 8
    return {"exact": bool(shape_ok and dtype_ok and hash_ok and all(exact_by_device)),
            "exact_by_device": exact_by_device, "shape_ok": shape_ok,
            "dtype_ok": dtype_ok, "hash_ok": hash_ok,
            "wire_shape": list(actual_wire.shape), "control_shape": list(actual_control.shape),
            "wire_dtype": str(actual_wire.dtype), "control_dtype": str(actual_control.dtype),
            "actual_sha256": _hash(actual_wire, actual_control)}


def make_sharded_call(mesh, *, interpret=False):
    partition = jax.sharding.PartitionSpec("core")

    def local(source, ranges, index, error):
        wire, control = pallas_pack_final_chunk(source[0], ranges[0], index[0],
                                                 world_size=8, prior_error=error[0],
                                                 interpret=interpret)
        return wire[None], control[None]

    return jax.jit(jax.shard_map(local, mesh=mesh,
                                 in_specs=(partition,) * 4,
                                 out_specs=(partition, partition), check_vma=False))


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise RuntimeError("requires eight physical TPU devices")
    report = {
        "status": "running",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")},
        "devices": [{"id": device.id, "kind": device.device_kind} for device in devices],
        "fixture_cases": ["cross_tile", "next_chunk", "empty", "bounds_error", "rank_error",
                          "prior_error", "final_hbm_tile", "exhausted_chunk"],
        "plane_cases": [],
    }
    result_path = output / "production_packing.json"

    def save():
        result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    mesh = jax.sharding.Mesh(np.asarray(devices), ("core",))
    partition = jax.sharding.PartitionSpec("core")
    for planes in (32, 35):
        folder = output / f"planes_{planes}"
        folder.mkdir(exist_ok=False)
        payload, intervals, chunks, prior = make_inputs(planes=planes)
        case = {"planes": planes, "status": "lowering", "scenarios": []}
        report["plane_cases"].append(case)
        save()
        call = make_sharded_call(mesh)
        inputs = tuple(jax.device_put(value, jax.sharding.NamedSharding(mesh, partition))
                       for value in (payload, intervals, chunks, prior))
        lowered = call.lower(*inputs)
        (folder / "lowered.mlir").write_text(lowered.as_text(), encoding="utf-8")
        case["status"] = "compiling"
        save()
        executable = lowered.compile()
        (folder / "compiled.hlo.txt").write_text(executable.as_text(), encoding="utf-8")
        for scenario in ("mixed", "all_live"):
            payload, intervals, chunks, prior = make_inputs(planes=planes, scenario=scenario)
            want_wire, want_control = expected(payload, intervals, chunks, prior)
            row = {"scenario": scenario, "status": "executing",
                   "input_sha256": _hash(payload, intervals, chunks, prior),
                   "expected_sha256": _hash(want_wire, want_control)}
            case["scenarios"].append(row)
            save()
            inputs = tuple(jax.device_put(value, jax.sharding.NamedSharding(mesh, partition))
                           for value in (payload, intervals, chunks, prior))
            actual_wire, actual_control = map(np.asarray, jax.block_until_ready(executable(*inputs)))
            row.update(validate_outputs(actual_wire, actual_control, want_wire, want_control))
            row["status"] = "complete"
            if row["shape_ok"]:
                row.update(max_abs_wire=int(np.max(np.abs(actual_wire.astype(np.int64) - want_wire.astype(np.int64)))),
                           max_abs_control=int(np.max(np.abs(actual_control.astype(np.int64) - want_control.astype(np.int64)))))
            save()
            if not row["exact"]:
                raise AssertionError(row)
        case["status"] = "complete"
        save()
    report["status"] = "complete"
    report["all_exact"] = True
    save()
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
