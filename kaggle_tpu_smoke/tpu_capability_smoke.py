import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")

# Kaggle's image can contain a much newer JAX than its bundled TPU runtime.
# Refresh libtpu before importing JAX so Pallas/Mosaic sees a compatible build.
libtpu_upgrade = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "libtpu"],
    text=True,
    capture_output=True,
)

RESULT = {"tests": {}}


def record(name, fn):
    started = time.perf_counter()
    try:
        value = fn()
        RESULT["tests"][name] = {
            "ok": True,
            "seconds": time.perf_counter() - started,
            "value": value,
        }
    except Exception as exc:
        RESULT["tests"][name] = {
            "ok": False,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    print("TEST", name, json.dumps(RESULT["tests"][name], default=str), flush=True)


import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

RESULT["environment"] = {
    "python": platform.python_version(),
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "device_count": jax.device_count(),
    "local_device_count": jax.local_device_count(),
    "devices": [str(x) for x in jax.devices()],
    "libtpu_upgrade_returncode": libtpu_upgrade.returncode,
    "libtpu_upgrade_stderr_tail": libtpu_upgrade.stderr[-1000:],
}
print("ENV", json.dumps(RESULT["environment"]), flush=True)


def test_uint64():
    @jax.jit
    def mix(x):
        x = x ^ (x >> jnp.uint64(30))
        x = x * jnp.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> jnp.uint64(27))
        return x

    x = jnp.arange(4096, dtype=jnp.uint64)
    y = mix(x).block_until_ready()
    return {"dtype": str(y.dtype), "checksum": int(jnp.bitwise_xor.reduce(y))}


record("jax_uint64_hash_ops", test_uint64)


def test_pallas_vector():
    from jax.experimental import pallas as pl

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = (x ^ jnp.uint32(0x9E3779B9)) + jnp.uint32(17)

    call = jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.uint32),
            in_specs=[pl.BlockSpec((8, 128), lambda: (0, 0))],
            out_specs=pl.BlockSpec((8, 128), lambda: (0, 0)),
            grid=(),
            name="beam_vector_smoke",
        )
    )
    x = jnp.arange(1024, dtype=jnp.uint32).reshape(8, 128)
    y = call(x).block_until_ready()
    expected = (x ^ jnp.uint32(0x9E3779B9)) + jnp.uint32(17)
    return {"equal": bool(jnp.array_equal(y, expected)), "shape": list(y.shape)}


record("pallas_vector", test_pallas_vector)


def test_pallas_matmul():
    from jax.experimental import pallas as pl

    shape_x = (256, 512)
    shape_w = (512, 512)

    def kernel(x_ref, w_ref, o_ref):
        o_ref[...] = jax.lax.dot_general(
            x_ref[...],
            w_ref[...],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    call = jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((256, 512), jnp.float32),
            in_specs=[
                pl.BlockSpec(shape_x, lambda: (0, 0)),
                pl.BlockSpec(shape_w, lambda: (0, 0)),
            ],
            out_specs=pl.BlockSpec((256, 512), lambda: (0, 0)),
            grid=(),
            name="beam_bf16_matmul_smoke",
        )
    )
    x = jnp.ones(shape_x, dtype=jnp.bfloat16)
    w = jnp.ones(shape_w, dtype=jnp.bfloat16)
    y = call(x, w).block_until_ready()
    return {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "first": float(y[0, 0]),
        "correct": bool(jnp.all(y == jnp.float32(512))),
    }


record("pallas_bf16_matmul", test_pallas_matmul)


def test_collective_histogram():
    n = jax.local_device_count()
    if n < 2:
        raise RuntimeError(f"need multiple local TPU cores, found {n}")

    def reduce_hist_impl(x):
        return jax.lax.psum(x, "core")

    reduce_hist = jax.pmap(reduce_hist_impl, axis_name="core")

    bins = 307201
    x = jnp.arange(n * bins, dtype=jnp.uint32).reshape(n, bins)
    first_started = time.perf_counter()
    y = reduce_hist(x).block_until_ready()
    first_seconds = time.perf_counter() - first_started
    steady_started = time.perf_counter()
    y = reduce_hist(x).block_until_ready()
    steady_seconds = time.perf_counter() - steady_started
    expected = jnp.sum(x, axis=0)
    return {
        "cores": n,
        "equal": bool(jnp.all(y[0] == expected)),
        "bins": bins,
        "bytes_per_core": bins * 4,
        "compile_and_first_seconds": first_seconds,
        "steady_seconds": steady_seconds,
    }


record("eight_core_psum", test_collective_histogram)


def test_lexicographic_sort():
    count = 1 << 18
    idx = jnp.arange(count, dtype=jnp.uint32)
    hi = ((idx * jnp.uint32(2654435761)) ^ (idx >> 3)).astype(jnp.uint32)
    lo = ((idx * jnp.uint32(2246822519)) ^ (idx >> 7)).astype(jnp.uint32)
    score = (idx % jnp.uint32(307201)).astype(jnp.uint32)

    @jax.jit
    def sort_pairs(a, b, c):
        return jax.lax.sort((a, b, c), dimension=0, num_keys=2, is_stable=True)

    compile_started = time.perf_counter()
    out = sort_pairs(hi, lo, score)
    out[0].block_until_ready()
    compile_seconds = time.perf_counter() - compile_started
    run_started = time.perf_counter()
    out = sort_pairs(hi, lo, score)
    out[0].block_until_ready()
    run_seconds = time.perf_counter() - run_started
    ordered = jnp.all(
        (out[0][1:] > out[0][:-1])
        | ((out[0][1:] == out[0][:-1]) & (out[1][1:] >= out[1][:-1]))
    )
    return {
        "items": count,
        "compile_and_first_seconds": compile_seconds,
        "steady_seconds": run_seconds,
        "ordered": bool(ordered),
    }


record("jax_hash_pair_sort", test_lexicographic_sort)


def inspect_megaminx_model():
    import torch

    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    matches = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob("weights_megaminx2048_512_8_e4000.pth"))
    if not matches:
        matches = list(Path("/kaggle/input").rglob("*.pth"))
    if not matches:
        raise FileNotFoundError("no attached .pth model found")
    path = matches[0]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("model", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj.state_dict()
    shapes = {str(k): list(v.shape) for k, v in state.items() if hasattr(v, "shape")}
    linear = {k: v for k, v in shapes.items() if k.endswith("weight") and len(v) == 2}
    return {"path": str(path), "linear_weight_shapes": linear, "tensor_count": len(shapes)}


record("megaminx_model_inspection", inspect_megaminx_model)

out = Path("/kaggle/working/tpu_capability_results.json")
out.write_text(json.dumps(RESULT, indent=2, default=str), encoding="utf-8")
print("RESULT_PATH", out, flush=True)
print("SUMMARY", json.dumps({k: v["ok"] for k, v in RESULT["tests"].items()}), flush=True)
