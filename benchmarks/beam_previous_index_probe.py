"""Physical alternatives to Mosaic-unsupported uint32 previous-index max."""
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

from benchmarks.beam_dedup_stage_probe import _numpy_stages
from tpu_beam_search.beam_dedup import _columns, _sort


CASES = ('maximum_control', 'where_candidate', 'arithmetic_candidate')


def _call(words, payload, count, threshold, *, variant, interpret):
    n = words.shape[1]

    def kernel(w, p, c, t, out):
        indices = jnp.arange(n, dtype=jnp.uint32)
        valid = ((indices < c[0]) & (w[6, :] <= t[0])).astype(jnp.uint32)
        data = jnp.concatenate((w[...], p[...], valid[None, :], indices[None, :]), axis=0)
        data = _sort(data, (9, 3, 2, 1, 0, 6, 8))
        if variant == 'maximum_control':
            previous_indices = jnp.maximum(indices, jnp.uint32(1)) - jnp.uint32(1)
        elif variant == 'where_candidate':
            previous_indices = jnp.where(indices == 0, jnp.uint32(0), indices - jnp.uint32(1))
        elif variant == 'arithmetic_candidate':
            previous_indices = indices - (indices != 0).astype(jnp.uint32)
        previous = _columns(data[:4], previous_indices)
        unique = (data[9] != 0) & ((indices == 0) | jnp.any(data[:4] != previous, axis=0))
        out[...] = jnp.concatenate((data[:9], unique[None, :].astype(jnp.uint32),
                                    indices[None, :]), axis=0)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((11, n), jnp.uint32),
        in_specs=tuple(pl.BlockSpec(x.shape) for x in (words, payload, count, threshold)),
        out_specs=pl.BlockSpec((11, n)), grid=(), interpret=interpret,
        name='beam_previous_index_' + variant,
    )(words, payload, count, threshold)


def build_cases(*, interpret=False):
    rng = np.random.default_rng(7517)
    n = 128
    words = rng.integers(0, 2**32, (8, n), dtype=np.uint32)
    ids = np.arange(n, dtype=np.uint32) % 17
    words[:4] = np.stack((ids, ids * 3, ids * 7, ids * 11))
    words[6] %= 7
    payload = np.arange(n, dtype=np.uint32)[None, :]
    count = np.array([125], np.uint32)
    threshold = np.array([5], np.uint32)
    expected = _numpy_stages(words, payload, count, threshold)['uniqueness']
    return [dict(name=name,
                 fn=lambda w, p, c, t, name=name: _call(
                     w, p, c, t, variant=name, interpret=interpret),
                 args=(words, payload, count, threshold), expected=expected)
            for name in CASES]


def _run(case, output):
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != 'tpu' for device in devices):
        raise RuntimeError('requires exactly eight real TPU devices')
    report = dict(scope='previous-index full uniqueness boundary; not performance',
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                  libtpu=importlib.metadata.version('libtpu'),
                  devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
                  case=case['name'], status='started', exact=False,
                  input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in case['args']])
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'previous_index_probe.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        replicated = jax.tree.map(lambda x: np.stack([x] * 8), case['args'])
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, 'core'),
                                             jax.sharding.PartitionSpec('core'))
        inputs = jax.tree.map(lambda x: jax.device_put(x, sharding), replicated)
        executable = jax.pmap(case['fn'], devices=devices).lower(*inputs).compile()
        started = time.perf_counter()
        actual = np.asarray(jax.block_until_ready(executable(*inputs)))
        report['execute_seconds'] = time.perf_counter() - started
        (output / (case['name'] + '.hlo.txt')).write_text(executable.as_text(), encoding='utf-8')
        expected = np.stack([case['expected']] * 8)
        report['mismatched_elements'] = int(np.count_nonzero(actual != expected))
        report['exact'] = report['mismatched_elements'] == 0 and actual.dtype == expected.dtype
        report['status'] = 'exact' if report['exact'] else 'mismatch'
    except Exception:
        report['status'] = 'error'
        report['error'] = traceback.format_exc()
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.case:
        _run(next(c for c in build_cases() if c['name'] == args.case), args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = {'scope': 'isolated previous-index probes', 'cases': []}
    for name in CASES:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'benchmarks.beam_previous_index_probe',
                   '--output', str(folder), '--case', name]
        with (folder / 'process.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row = {'name': name, 'returncode': result.returncode}
        result_path = folder / 'previous_index_probe.json'
        if result_path.exists():
            row['report'] = json.loads(result_path.read_text(encoding='utf-8'))
        bundle['cases'].append(row)
        (args.output / 'previous_index_bundle.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
