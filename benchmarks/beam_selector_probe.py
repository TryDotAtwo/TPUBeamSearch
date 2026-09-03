"""Minimal physical probe for the survivor selector rejected by Mosaic.

Each case performs the same uint32 [8,128] selection.  The variants separate
predicate shape, boolean conversion, and data selection; each must run in its
own subprocess because a compiler failure may abort natively.
"""
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


CASE_NAMES = (
    'where_broadcast_bool',
    'where_full_bool',
    'arithmetic_broadcast_bool',
    'arithmetic_full_bool',
    'arithmetic_u32_mask',
)


def _call(data, neutral, mask, *, variant, interpret):
    def kernel(d, z, m, out):
        data_value = d[...]
        neutral_value = z[...]
        mask_value = m[...]
        if variant == 'where_broadcast_bool':
            value = jnp.where(mask_value != 0, data_value, neutral_value)
        elif variant == 'where_full_bool':
            predicate = jnp.broadcast_to(mask_value != 0, data_value.shape)
            value = jnp.where(predicate, data_value, neutral_value)
        elif variant == 'arithmetic_broadcast_bool':
            indicator = (mask_value != 0).astype(jnp.uint32)
            value = indicator * data_value + (jnp.uint32(1) - indicator) * neutral_value
        elif variant == 'arithmetic_full_bool':
            indicator = jnp.broadcast_to(mask_value != 0, data_value.shape).astype(jnp.uint32)
            value = indicator * data_value + (jnp.uint32(1) - indicator) * neutral_value
        elif variant == 'arithmetic_u32_mask':
            value = mask_value * data_value + (jnp.uint32(1) - mask_value) * neutral_value
        else:
            raise AssertionError(variant)
        out[...] = value

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(data.shape, jnp.uint32),
        in_specs=tuple(pl.BlockSpec(x.shape) for x in (data, neutral, mask)),
        out_specs=pl.BlockSpec(data.shape),
        grid=(),
        interpret=interpret,
        name='beam_selector_probe_' + variant,
    )(data, neutral, mask)


def build_cases(*, interpret=False):
    rng = np.random.default_rng(7017)
    data = rng.integers(0, 2**32, (8, 128), dtype=np.uint32)
    neutral = np.zeros_like(data)
    neutral[6] = np.uint32(0xffffffff)
    mask = ((np.arange(128) % 3) != 0).astype(np.uint32)[None, :]
    expected = mask * data + (np.uint32(1) - mask) * neutral
    return [dict(name=name,
                 fn=lambda d, z, m, name=name: _call(d, z, m, variant=name, interpret=interpret),
                 args=(data, neutral, mask), expected=expected)
            for name in CASE_NAMES]


def _run_case(case, output):
    devices = jax.devices()
    if len(devices) != 8 or any(device.platform != 'tpu' for device in devices):
        raise RuntimeError('requires exactly eight real TPU devices')
    report = dict(scope='minimal selector lowering probe; not beam performance',
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                  libtpu=importlib.metadata.version('libtpu'),
                  devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
                  case=case['name'], status='started', exact=False,
                  input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in case['args']])
    path = output / 'selector_probe.json'
    output.mkdir(parents=True, exist_ok=True)
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
    except Exception:
        report['status'] = 'error'
        report['error'] = traceback.format_exc()
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASE_NAMES)
    args = parser.parse_args()
    if args.case:
        case = next(case for case in build_cases() if case['name'] == args.case)
        _run_case(case, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {'scope': 'isolated selector probes', 'cases': []}
    for name in CASE_NAMES:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'benchmarks.beam_selector_probe',
                   '--output', str(folder), '--case', name]
        with (folder / 'process.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row = {'name': name, 'returncode': result.returncode}
        result_path = folder / 'selector_probe.json'
        if result_path.exists():
            row['report'] = json.loads(result_path.read_text(encoding='utf-8'))
        aggregate['cases'].append(row)
        (args.output / 'selector_probe_bundle.json').write_text(
            json.dumps(aggregate, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
