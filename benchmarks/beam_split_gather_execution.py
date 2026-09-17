"""Physical TPU execution gate for the split-gather diagnostic."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

from benchmarks.beam_response_isolation_probe import stage_call


def expected(data, ranges, prior):
    out = np.zeros((8, 32, 128), np.uint32)
    if prior[0, 0]:
        return out
    for peer in range(8):
        start, count = map(int, ranges[:2, peer])
        if not count:
            continue
        length = min(count, 128)
        aligned = start // 128 * 128
        shift = start % 128
        tile = np.zeros((32, 256), np.uint32)
        tile[:, :128] = data[:, aligned:aligned + 128]
        if shift + length > 128:
            tile[:, 128:] = data[:, aligned + 128:aligned + 256]
        positions = np.arange(128) + shift
        low = tile[:, (positions % 128)]
        high = tile[:, 128 + (positions % 128)]
        out[peer] = np.where(positions[None, :] < 128, low, high)
    return out


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = np.arange(8 * 32 * 2048, dtype=np.uint32).reshape(8, 32, 2048)
    ranges = np.zeros((8, 3, 128), np.uint32)
    ranges[:, 0, :8] = np.array([1, 127, 128, 129, 255, 256, 1920, 2048])
    ranges[:, 1, :8] = np.array([128, 2, 128, 0, 1, 129, 128, 0])
    prior = np.zeros((8, 1, 128), np.uint32)
    index = jnp.array([0], jnp.uint32)
    call, _ = stage_call("packing_split_gather", SimpleNamespace(size=8), interpret=False)
    actual, control = jax.block_until_ready(call(
        jnp.asarray(data), jnp.asarray(ranges[0]), jnp.asarray(prior[0]), index))
    actual = np.asarray(actual)
    control = np.asarray(control)
    want = expected(data[0], ranges[0], prior[0])
    result = {
        "devices": [{"id": d.id, "kind": d.device_kind} for d in jax.devices()],
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact": bool(np.array_equal(actual, want)),
        "max_abs": int(np.max(np.abs(actual.astype(np.int64) - want.astype(np.int64)))),
        "control_nonzero": int(np.count_nonzero(control)),
    }
    (output / "split_gather_execution.json").write_text(json.dumps(result, indent=2))
    if not result["exact"]:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
