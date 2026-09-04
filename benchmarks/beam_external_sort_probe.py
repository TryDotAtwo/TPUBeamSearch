"""Physical eight-TPU gate for the first N=256 HBM-staged sort."""
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

from tpu_beam_search.beam_external_sort import pallas_external_bitonic_sort


def expected_sort(data):
    order = sorted(range(data.shape[1]), key=lambda i: (
        1 - int(data[9, i]),
        int(data[3, i]), int(data[2, i]), int(data[1, i]), int(data[0, i]),
        int(data[6, i]), int(data[8, i]), int(data[10, i])))
    return data[:, order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires exactly eight real TPU devices')
    ranks, capacity = 8, 256
    rng = np.random.default_rng(20260905)
    per_rank = rng.integers(0, 2**32, (ranks, 11, capacity), dtype=np.uint32)
    per_rank[:, 9] = (np.arange(capacity)[None, :] % 7 != 0).astype(np.uint32)
    per_rank[:, 10] = np.arange(capacity, dtype=np.uint32)[None, :]
    # Force duplicates and opposed payload order across the 128-column boundary.
    per_rank[:, :4, 129] = per_rank[:, :4, 3]
    per_rank[:, 6, 129] = per_rank[:, 6, 3] - np.uint32(1)
    expected = np.stack([expected_sort(value) for value in per_rank])
    global_input = per_rank.transpose(1, 0, 2).reshape(11, ranks * capacity)
    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    partition = jax.sharding.PartitionSpec(None, 'core')
    sharding = jax.sharding.NamedSharding(mesh, partition)
    placed = jax.device_put(global_input, sharding)

    def local_sort(value):
        return pallas_external_bitonic_sort(
            value, key_planes=(9, 3, 2, 1, 0, 6, 8, 10),
            validity_plane=9, tile_candidates=128)

    fn = jax.jit(jax.shard_map(
        local_sort, mesh=mesh, in_specs=partition, out_specs=partition,
        check_vma=False))
    started = time.perf_counter()
    executable = fn.lower(placed).compile()
    compile_seconds = time.perf_counter() - started
    (args.output / 'external_sort_256.hlo.txt').write_text(
        executable.as_text(), encoding='utf-8')
    actual_global = np.asarray(jax.block_until_ready(executable(placed)))
    actual = actual_global.reshape(11, ranks, capacity).transpose(1, 0, 2)
    mismatches = int(np.count_nonzero(actual != expected))
    for _ in range(3):
        jax.block_until_ready(executable(placed))
    samples = []
    for _ in range(21):
        start = time.perf_counter_ns()
        jax.block_until_ready(executable(placed))
        samples.append((time.perf_counter_ns() - start) / 1e6)
    report = dict(
        exact=mismatches == 0, mismatched_elements=mismatches,
        capacity=capacity, tile_candidates=128, runs=2,
        output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        compile_seconds=compile_seconds,
        source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id, kind=d.device_kind,
                      process_index=d.process_index) for d in devices],
        timing=dict(warmup=3, repeats=21, samples_ms=samples,
                    median_ms=float(np.median(samples)),
                    p10_ms=float(np.percentile(samples, 10)),
                    p90_ms=float(np.percentile(samples, 90))),
        scope='N=256 HBM-staged global bitonic sort gate; not dedup/beam',
    )
    (args.output / 'beam_external_sort_probe.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not report['exact']:
        raise RuntimeError('external sort mismatch')


if __name__ == '__main__':
    main()
