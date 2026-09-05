"""Eight-TPU external Stream3 dedup gate; no routing or beam performance claim."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.beam_external_sort import pallas_external_stream3_dedup


def oracle(words, payload, count, threshold):
    order = sorted((i for i in range(count) if words[6, i] <= threshold),
                   key=lambda i: tuple(int(words[p, i]) for p in (3, 2, 1, 0))
                   + (int(words[6, i]), int(payload[0, i])))
    seen, selected = set(), []
    for i in order:
        key = tuple(words[:4, i])
        if key not in seen:
            seen.add(key)
            selected.append(i)
    result = np.zeros_like(words)
    result[6] = np.uint32(0xffffffff)
    result[:, :len(selected)] = words[:, selected]
    counts = np.zeros((1, 128), np.uint32)
    counts[0, 0] = len(selected)
    return result, counts


def digest(arrays):
    h = hashlib.sha256()
    for value in arrays:
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def local_dedup(w, v, c, t):
    result, amount = pallas_external_stream3_dedup(w[0], v[0], c[0, 0], t[0, 0])
    return result[None], amount[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires exactly eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id, kind=d.device_kind) for d in devices], cases=[],
        scope='independent eight-device external S3 dedup, before owner routing')
    (output / 'external_dedup.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    p = jax.sharding.PartitionSpec('core', None, None)
    sharding = jax.sharding.NamedSharding(mesh, p)
    for n in (256, 512):
        seed = 20260905 + n
        rng = np.random.default_rng(seed)
        words = rng.integers(0, 2**32, (8, 8, n), dtype=np.uint32)
        words[:, :4] = 0
        words[:, 0] = np.arange(n, dtype=np.uint32)[None, :]
        words[:, 0, 128] = 127
        words[:, 6] = rng.integers(0, 21, (8, n), dtype=np.uint32)
        words[:, 6, 127:129] = 5
        words[2, :4] = 0  # all duplicate, including hash zero
        payload = np.broadcast_to(np.arange(n, dtype=np.uint32), (8, 1, n)).copy()
        counts = np.array([0, n, n, n - 1, 129, 128, 1, n], np.uint32)
        thresholds = np.array([10, 20, 20, 10, 5, 0xffffffff, 0, 10], np.uint32)
        expected = [oracle(w, v, int(c), int(t))
                    for w, v, c, t in zip(words, payload, counts, thresholds)]
        args = tuple(jax.device_put(x, sharding) for x in
                     (words, payload, counts[:, None, None], thresholds[:, None, None]))

        fn = jax.jit(jax.shard_map(local_dedup, mesh=mesh, in_specs=(p, p, p, p),
                                  out_specs=(p, p), check_vma=False))
        print(json.dumps(dict(event='compile_start', capacity=n, seed=seed,
                              input_sha256=digest((words, payload, counts, thresholds)))),
              flush=True)
        start = time.perf_counter()
        executable = fn.lower(*args).compile()
        compile_s = time.perf_counter() - start
        (output / f'dedup_{n}.hlo.txt').write_text(executable.as_text())
        actual = tuple(np.asarray(x) for x in jax.block_until_ready(executable(*args)))
        reference = tuple(np.stack([e[i] for e in expected]) for i in range(2))
        mismatch = sum(int(np.count_nonzero(a != e)) for a, e in zip(actual, reference))
        case = dict(capacity=n, seed=seed, mismatches=mismatch,
                    input_sha256=digest((words, payload, counts, thresholds)),
                    output_sha256=digest(actual), expected_sha256=digest(reference),
                    compile_seconds=compile_s, exact=mismatch == 0)
        if mismatch == 0:
            for _ in range(3):
                jax.block_until_ready(executable(*args))
            samples = []
            for _ in range(21):
                start = time.perf_counter_ns()
                jax.block_until_ready(executable(*args))
                samples.append((time.perf_counter_ns() - start) / 1e6)
            case['timing'] = dict(warmup=3, repeats=21, samples_ms=samples,
                                 median_ms=float(np.median(samples)),
                                 p10_ms=float(np.percentile(samples, 10)),
                                 p90_ms=float(np.percentile(samples, 90)))
        report['cases'].append(case)
        (output / 'external_dedup.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(case), flush=True)
        if mismatch:
            raise RuntimeError('external dedup mismatch')


if __name__ == '__main__':
    main()
