"""Physical bisection of the first real bitonic compare/exchange."""
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

from tpu_beam_search.beam_dedup import _columns


CASES = ('partner_gather', 'swap_predicate', 'select_broadcast',
         'select_full', 'select_rowwise', 'select_arithmetic',
         'swap_boolean_logic', 'select_boolean_logic')
KEYS = (9, 3, 2, 1, 0, 6, 8)


def _call(source, *, variant, interpret):
    n = source.shape[1]
    shape = (1, n) if variant in ('swap_predicate', 'swap_boolean_logic') else source.shape

    def kernel(ref, out):
        data = ref[...]
        indices = jnp.arange(n, dtype=jnp.uint32)
        partner = _columns(data, indices ^ 1)
        if variant == 'partner_gather':
            out[...] = partner
            return
        less = jnp.zeros((n,), jnp.bool_)
        equal = jnp.ones((n,), jnp.bool_)
        for plane in KEYS:
            a, b = data[plane], partner[plane]
            if plane == 9:
                a, b = 1 - a, 1 - b
            less = less | (equal & (a < b))
            equal = equal & (a == b)
        want_min = ((indices & 2) == 0) == ((indices & 1) == 0)
        if variant in ('swap_boolean_logic', 'select_boolean_logic'):
            swap = (want_min & ~less & ~equal) | (~want_min & less)
        else:
            swap = jnp.where(want_min, ~less & ~equal, less)
        if variant in ('swap_predicate', 'swap_boolean_logic'):
            out[...] = swap[None, :].astype(jnp.uint32)
        elif variant == 'select_broadcast':
            out[...] = jnp.where(swap[None, :], partner, data)
        elif variant == 'select_full':
            out[...] = jnp.where(jnp.broadcast_to(swap[None, :], data.shape), partner, data)
        elif variant == 'select_rowwise':
            out[...] = jnp.stack([jnp.where(swap, partner[row], data[row])
                                  for row in range(data.shape[0])])
        elif variant == 'select_arithmetic':
            indicator = swap.astype(jnp.uint32)[None, :]
            out[...] = indicator * partner + (jnp.uint32(1) - indicator) * data
        elif variant == 'select_boolean_logic':
            out[...] = jnp.where(swap[None, :], partner, data)

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape, jnp.uint32),
                          in_specs=(pl.BlockSpec(source.shape),), out_specs=pl.BlockSpec(shape),
                          grid=(), interpret=interpret,
                          name='beam_compare_exchange_' + variant)(source)


def _numpy_expected(source):
    n = source.shape[1]
    indices = np.arange(n)
    partner = source[:, indices ^ 1]
    less = np.zeros(n, bool)
    equal = np.ones(n, bool)
    for plane in KEYS:
        a, b = source[plane], partner[plane]
        if plane == 9:
            a, b = 1 - a, 1 - b
        less |= equal & (a < b)
        equal &= a == b
    want_min = ((indices & 2) == 0) == ((indices & 1) == 0)
    swap = np.where(want_min, ~less & ~equal, less)
    selected = np.where(swap[None, :], partner, source)
    return partner, swap.astype(np.uint32)[None, :], selected


def build_cases(*, interpret=False):
    rng = np.random.default_rng(7331)
    source = rng.integers(0, 2**32, (11, 128), dtype=np.uint32)
    source[9] = (np.arange(128) % 5 != 0).astype(np.uint32)
    source[10] = np.arange(128, dtype=np.uint32)
    partner, predicate, selected = _numpy_expected(source)
    expected = {'partner_gather': partner, 'swap_predicate': predicate,
                'select_broadcast': selected, 'select_full': selected,
                'select_rowwise': selected, 'select_arithmetic': selected,
                'swap_boolean_logic': predicate, 'select_boolean_logic': selected}
    return [dict(name=name,
                 fn=lambda x, name=name: _call(x, variant=name, interpret=interpret),
                 args=(source,), expected=expected[name]) for name in CASES]


def _run(case, output):
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != 'tpu' for device in devices):
        raise RuntimeError('requires exactly eight real TPU devices')
    report = dict(scope='one real compare/exchange lowering probe; not performance',
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                  libtpu=importlib.metadata.version('libtpu'),
                  devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
                  case=case['name'], status='started', exact=False,
                  input_sha256=hashlib.sha256(case['args'][0].tobytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'compare_exchange_probe.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        replicated = np.stack([case['args'][0]] * 8)
        sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(devices, 'core'),
                                             jax.sharding.PartitionSpec('core'))
        value = jax.device_put(replicated, sharding)
        fn = jax.pmap(case['fn'], devices=devices)
        start = time.perf_counter()
        executable = fn.lower(value).compile()
        report['compile_seconds'] = time.perf_counter() - start
        (output / (case['name'] + '.hlo.txt')).write_text(executable.as_text(), encoding='utf-8')
        actual = np.asarray(jax.block_until_ready(executable(value)))
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
    bundle = {'scope': 'isolated first compare/exchange probes', 'cases': []}
    for name in CASES:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'benchmarks.beam_compare_exchange_probe',
                   '--output', str(folder), '--case', name]
        with (folder / 'process.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row = {'name': name, 'returncode': result.returncode}
        result_path = folder / 'compare_exchange_probe.json'
        if result_path.exists():
            row['report'] = json.loads(result_path.read_text(encoding='utf-8'))
        bundle['cases'].append(row)
        (args.output / 'compare_exchange_bundle.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
