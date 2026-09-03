"""Physical eight-TPU compile/correctness bundle; not a whole-beam benchmark."""
import argparse
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.beam_transport import pallas_pack_candidates
from tpu_beam_search.beam_stream2 import pallas_hash_goal
from tpu_beam_search.beam_hash import pallas_route_hashes
from tpu_beam_search.beam_dedup import pallas_threshold_dedup


def _distribution(lo, hi, salt):
    mask = 2**64 - 1
    def mix(x):
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    return mix(lo ^ (((hi << 32) | (hi >> 32)) & mask) ^ salt
               ^ mix((hi + 0x9e3779b97f4a7c15) & mask))


def build_cases(*, interpret=False):
    rng = np.random.default_rng(9341)
    cases = []
    words = rng.integers(0, 2**32, (8, 65536), dtype=np.uint32)
    for pipelined in (False, True):
        for buffers in (2, 3):
            cases.append(dict(name=f'pack_{"pipeline" if pipelined else "serial"}_b{buffers}',
                fn=functools.partial(pallas_pack_candidates, buffer_count=buffers,
                                     pipelined=pipelined, interpret=interpret),
                args=(words[:4], words[4:6], words[6:7], words[7:8]), expected=words))
    hashes = words[:4, :256].copy()
    expected = np.zeros((2, 256), np.uint32)
    for i in range(256):
        lo = int(hashes[0, i]) | int(hashes[1, i]) << 32
        hi = int(hashes[2, i]) | int(hashes[3, i]) << 32
        expected[:, i] = [_distribution(lo, hi, 0x243f6a8885a308d3) % 8,
                          _distribution(lo, hi, 0x13198a2e03707344) % 7]
    cases.append(dict(name='route_8_7', fn=functools.partial(pallas_route_hashes,
        world_size=8, shard_count=7, interpret=interpret), args=(hashes,), expected=expected))
    for logical, moves, storage in ((120, 24, 128), (150, 30, 160)):
        batch, count, classes = 8, 7, 12
        parents = np.zeros((batch, storage), np.uint8)
        parents[:, :logical] = rng.integers(0, classes, (batch, logical), dtype=np.uint8)
        generators = np.tile(np.arange(storage, dtype=np.int32), (moves, 1))
        for m in range(moves):
            generators[m, :logical] = rng.permutation(logical)
        central = parents[1, generators[-1]].copy()
        table = rng.integers(0, 2**32, (4, storage, classes), dtype=np.uint32)
        table[:, logical:] = 0
        capacity = ((batch * moves + 127) // 128) * 128
        h = np.zeros((4, capacity), np.uint32)
        g = np.zeros((1, capacity), np.uint32)
        valid = np.zeros((1, capacity), np.uint32)
        for p in range(count):
            for m in range(moves):
                child = parents[p, generators[m]]
                i = p * moves + m
                h[:, i] = np.bitwise_xor.reduce(table[:, np.arange(storage), child], axis=1)
                g[0, i] = np.array_equal(child, central)
                valid[0, i] = 1
        cases.append(dict(name=f'hash_goal_{logical}_{moves}',
            fn=functools.partial(pallas_hash_goal, interpret=interpret),
            args=(parents, generators, central, table.reshape(4, -1), np.array([count], np.uint32)),
            expected=(h, g, valid)))
    for n in (128, 256):
        w = rng.integers(0, 2**32, (8, n), dtype=np.uint32)
        ids = np.arange(n, dtype=np.uint32) % 11
        w[:4] = np.stack((ids, ids * 3, ids * 7, ids * 11))
        w[6] %= 5
        payload = np.arange(n, dtype=np.uint32)[None, :]
        for mode in ('stream3', 'stream4'):
            keys = lambda i: tuple(int(w[k, i]) for k in (3, 2, 1, 0, 6)) + (
                (i,) if mode == 'stream3' else tuple(int(w[k, i]) for k in (5, 4, 7)))
            order = sorted((i for i in range(n - 3) if w[6, i] <= 3), key=keys)
            selected, seen = [], set()
            for i in order:
                identity = tuple(int(x) for x in w[:4, i])
                if identity not in seen:
                    seen.add(identity)
                    selected.append(i)
            out = np.zeros_like(w)
            out[6] = 0xffffffff
            out[:, :len(selected)] = w[:, selected]
            cases.append(dict(name=f'dedup_{mode}_{n}',
                fn=functools.partial(pallas_threshold_dedup, mode=mode, interpret=interpret),
                args=(w, payload, np.array([n - 3], np.uint32), np.array([3], np.uint32)),
                expected=(out, np.array([len(selected)], np.uint32))))
    return cases


def compare_outputs(actual, expected):
    a, ta = jax.tree.flatten(actual)
    b, tb = jax.tree.flatten(expected)
    if ta != tb or any(np.shape(x) != np.shape(y) for x, y in zip(a, b)):
        return dict(exact=False, mismatched_elements=None, structure_mismatch=True)
    mismatches = sum(int(np.count_nonzero(np.asarray(x) != np.asarray(y))) for x, y in zip(a, b))
    dtype_match = all(np.asarray(x).dtype == np.asarray(y).dtype for x, y in zip(a, b))
    return dict(exact=mismatches == 0 and dtype_match, mismatched_elements=mismatches,
                dtype_match=dtype_match,
                output_sha256=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in a])


def measure_interleaved(variants, *, warmup=3, repeats=21):
    if not variants or warmup < 0 or repeats <= 0:
        raise ValueError('requires variants, nonnegative warmup and positive repeats')
    names = list(variants)
    for _ in range(warmup):
        for name in names:
            jax.block_until_ready(variants[name]())
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else names[::-1]
        for name in order:
            start = time.perf_counter_ns()
            jax.block_until_ready(variants[name]())
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    return {name: dict(samples_ms=s, median_ms=float(np.median(s)),
                       p10_ms=float(np.percentile(s, 10)), p90_ms=float(np.percentile(s, 90)))
            for name, s in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires exactly eight real TPU devices')
    report = dict(scope='eight-device primitive correctness and synchronized timing, NOT complete beam or overlap proof',
        source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id, kind=d.device_kind, process_index=d.process_index) for d in devices],
        x64_enabled=bool(jax.config.jax_enable_x64), cases=[], timings={},
        timing_protocol=dict(warmup=3, repeats=21, order='alternating forward/reverse',
                             scope='complete primitive call on eight devices; no placement/compile'))
    path = args.output / 'beam_primitives.json'
    def save():
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    eligible = {}
    for case in build_cases():
        row = dict(name=case['name'], input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in case['args']],
                   status='started', phase='placement')
        report['cases'].append(row)
        save()
        print('START', case['name'], flush=True)
        try:
            replicated = jax.tree.map(lambda x: np.stack([x] * 8), case['args'])
            sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, 'core'),
                                                 jax.sharding.PartitionSpec('core'))
            inputs = jax.tree.map(lambda x: jax.device_put(x, sharding), replicated)
            fn = jax.pmap(case['fn'], devices=devices)
            row['phase'] = 'compile'
            start = time.perf_counter()
            lowered = fn.lower(*inputs)
            executable = lowered.compile()
            row['compile_seconds'] = time.perf_counter() - start
            (args.output / (case['name'] + '.hlo.txt')).write_text(executable.as_text(), encoding='utf-8')
            row['phase'] = 'execute'
            actual = jax.block_until_ready(executable(*inputs))
            expected = jax.tree.map(lambda x: np.stack([x] * 8), case['expected'])
            row.update(compare_outputs(actual, expected))
            row['status'] = 'exact' if row['exact'] else 'mismatch'
            if row['exact']:
                eligible[case['name']] = functools.partial(executable, *inputs)
        except Exception:
            row['status'] = 'error'
            row['error'] = traceback.format_exc()
        save()
        print(row['name'], row['status'], flush=True)
    report['all_exact'] = all(r['status'] == 'exact' for r in report['cases'])
    for group, predicate in [('pack_matched', lambda name: name.startswith('pack_')),
                             ('other_diagnostic', lambda name: not name.startswith('pack_'))]:
        variants = {name: fn for name, fn in eligible.items() if predicate(name)}
        if not variants:
            continue
        try:
            print('TIMING', group, flush=True)
            report['timings'][group] = measure_interleaved(variants)
            if group == 'pack_matched':
                for value in report['timings'][group].values():
                    value['aggregate_candidates_per_second'] = 8 * 65536 / (value['median_ms'] / 1000)
                    value['candidates_per_device'] = 65536
        except Exception:
            report['timings'][group] = dict(error=traceback.format_exc())
        save()
    save()


if __name__ == '__main__':
    main()
