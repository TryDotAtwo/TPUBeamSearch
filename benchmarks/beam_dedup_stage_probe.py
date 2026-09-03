"""Bisect the real Stream3 dedup lowering pipeline on physical TPU."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from tpu_beam_search.beam_dedup import _columns, _sort


STAGES = ('initial', 'first_sort', 'uniqueness', 'second_sort', 'final_select')


def _pallas_stage(words, payload, count, threshold, *, stage, interpret):
    n = words.shape[1]
    output_shape = (8, n) if stage == 'final_select' else (11, n)

    def kernel(w, p, c, t, out):
        indices = jnp.arange(n, dtype=jnp.uint32)
        valid = ((indices < c[0]) & (w[6, :] <= t[0])).astype(jnp.uint32)
        data = jnp.concatenate((w[...], p[...], valid[None, :], indices[None, :]), axis=0)
        if stage == 'initial':
            out[...] = data
            return
        data = _sort(data, (9, 3, 2, 1, 0, 6, 8))
        if stage == 'first_sort':
            out[...] = data
            return
        previous = _columns(data[:4], jnp.maximum(indices, 1) - 1)
        unique = (data[9] != 0) & ((indices == 0) | jnp.any(data[:4] != previous, axis=0))
        data = jnp.concatenate((data[:9], unique[None, :].astype(jnp.uint32),
                                indices[None, :]), axis=0)
        if stage == 'uniqueness':
            out[...] = data
            return
        data = _sort(data, (9, 10))
        if stage == 'second_sort':
            out[...] = data
            return
        keep = data[9] != 0
        neutral = jnp.zeros((8, n), jnp.uint32).at[6, :].set(jnp.uint32(0xffffffff))
        out[...] = jnp.where(keep[None, :], data[:8], neutral)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, jnp.uint32),
        in_specs=tuple(pl.BlockSpec(x.shape) for x in (words, payload, count, threshold)),
        out_specs=pl.BlockSpec(output_shape), grid=(), interpret=interpret,
        name='beam_dedup_stage_' + stage,
    )(words, payload, count, threshold)


def _numpy_stages(words, payload, count, threshold):
    n = words.shape[1]
    index = np.arange(n, dtype=np.uint32)
    valid = ((index < count[0]) & (words[6] <= threshold[0])).astype(np.uint32)
    initial = np.concatenate((words, payload, valid[None, :], index[None, :]), axis=0)
    order1 = np.array(sorted(range(n), key=lambda i: (
        1 - int(initial[9, i]), int(initial[3, i]), int(initial[2, i]),
        int(initial[1, i]), int(initial[0, i]), int(initial[6, i]), int(initial[8, i]))))
    first = initial[:, order1]
    unique = np.zeros(n, np.uint32)
    for i in range(n):
        unique[i] = first[9, i] != 0 and (i == 0 or np.any(first[:4, i] != first[:4, i - 1]))
    uniqueness = np.concatenate((first[:9], unique[None, :], index[None, :]), axis=0)
    order2 = np.array(sorted(range(n), key=lambda i: (1 - int(uniqueness[9, i]),
                                                       int(uniqueness[10, i]))))
    second = uniqueness[:, order2]
    final = np.zeros((8, n), np.uint32)
    final[6] = np.uint32(0xffffffff)
    keep = second[9] != 0
    final[:, keep] = second[:8, keep]
    return dict(initial=initial, first_sort=first, uniqueness=uniqueness,
                second_sort=second, final_select=final)


def build_cases(*, interpret=False):
    rng = np.random.default_rng(7191)
    n = 128
    words = rng.integers(0, 2**32, (8, n), dtype=np.uint32)
    ids = np.arange(n, dtype=np.uint32) % 17
    words[:4] = np.stack((ids, ids * 3, ids * 7, ids * 11))
    words[6] %= 7
    payload = np.arange(n, dtype=np.uint32)[None, :]
    count = np.array([125], np.uint32)
    threshold = np.array([5], np.uint32)
    expected = _numpy_stages(words, payload, count, threshold)
    return [dict(name=stage,
                 fn=lambda w, p, c, t, stage=stage: _pallas_stage(
                     w, p, c, t, stage=stage, interpret=interpret),
                 args=(words, payload, count, threshold), expected=expected[stage])
            for stage in STAGES]


def _run_case(case, output):
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != 'tpu' for device in devices):
        raise RuntimeError('requires exactly eight real TPU devices')
    report = dict(scope='real dedup stage compile/correctness probe; not beam performance',
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                  libtpu=importlib.metadata.version('libtpu'),
                  devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
                  case=case['name'], status='started', exact=False,
                  input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in case['args']])
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'dedup_stage_probe.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        replicated = jax.tree.map(lambda x: np.stack([x] * 8), case['args'])
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, 'core'),
                                             jax.sharding.PartitionSpec('core'))
        inputs = jax.tree.map(lambda x: jax.device_put(x, sharding), replicated)
        fn = jax.pmap(case['fn'], devices=devices)
        start = time.perf_counter()
        executable = fn.lower(*inputs).compile()
        report['compile_seconds'] = time.perf_counter() - start
        (output / (case['name'] + '.hlo.txt')).write_text(executable.as_text(), encoding='utf-8')
        actual = np.asarray(jax.block_until_ready(executable(*inputs)))
        expected = np.stack([case['expected']] * 8)
        report['mismatched_elements'] = int(np.count_nonzero(actual != expected))
        report['exact'] = report['mismatched_elements'] == 0 and actual.dtype == expected.dtype
        report['status'] = 'exact' if report['exact'] else 'mismatch'
        report['output_sha256'] = hashlib.sha256(actual.tobytes()).hexdigest()
    except Exception:
        report['status'] = 'error'
        report['error'] = traceback.format_exc()
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=STAGES)
    args = parser.parse_args()
    if args.case:
        _run_case(next(c for c in build_cases() if c['name'] == args.case), args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {'scope': 'isolated real dedup stage probes', 'cases': []}
    for name in STAGES:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'benchmarks.beam_dedup_stage_probe',
                   '--output', str(folder), '--case', name]
        with (folder / 'process.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row = {'name': name, 'returncode': result.returncode}
        result_path = folder / 'dedup_stage_probe.json'
        if result_path.exists():
            row['report'] = json.loads(result_path.read_text(encoding='utf-8'))
        aggregate['cases'].append(row)
        (args.output / 'dedup_stage_bundle.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
