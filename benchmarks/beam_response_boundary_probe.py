"""Physical, isolated response byte-adapter and grouping correctness probe."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.beam_external_dedup_probe import digest
from benchmarks.beam_response_epoch_fixture import fixtures
from tpu_beam_search.beam_final_transport import pallas_wire_to_planes
from tpu_beam_search.beam_final_group import pallas_group_final_records


def host_inputs():
    name, (wire, requests, control, _) = next(
        entry for entry in fixtures() if entry[0] == 'self')
    assert name == 'self'
    ranks = np.zeros((8, 1, 2048), np.uint32)
    valid = np.zeros_like(ranks)
    for device in range(8):
        live = int(control[device, 0, 0])
        assert live == 129
        ranks[device, 0, :live] = requests[device, 3, :live] & np.uint32(65535)
        valid[device, 0, :live] = 1
    planes = wire.copy().view('<u4').reshape(8, 2048, 32).transpose(0, 2, 1).copy()
    return wire, planes, ranks, valid


def host_grouped(planes, ranks, valid):
    expected = np.empty((8, 35, 2048), np.uint32)
    for device in range(8):
        order = sorted(range(2048), key=lambda i: (
            0 if valid[device, 0, i] else 1,
            int(ranks[device, 0, i]), i))
        expected[device, :32] = planes[device][:, order]
        expected[device, 32] = ranks[device, 0, order]
        expected[device, 33] = order
        expected[device, 34] = valid[device, 0, order]
    return expected


def assess(actual, expected):
    actual, expected = map(np.asarray, (actual, expected))
    shape_ok = actual.shape == expected.shape
    dtype_ok = actual.dtype == expected.dtype == np.dtype('uint32')
    mismatches = ([int(np.count_nonzero(actual[i] != expected[i])) for i in range(8)]
                  if shape_ok else None)
    exact = bool(shape_ok and dtype_ok and not any(mismatches)
                 and digest((actual,)) == digest((expected,)))
    positions = np.argwhere(actual != expected)[:16] if shape_ok else []
    return dict(exact=exact, shape_ok=shape_ok, dtype_ok=dtype_ok,
                shape=list(actual.shape), dtype=str(actual.dtype),
                mismatches=mismatches, expected_sha256=digest((expected,)),
                output_sha256=digest((actual,)),
                first_mismatches=[[int(x) for x in (*p, actual[tuple(p)], expected[tuple(p)])]
                                  for p in positions])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(('git', 'rev-parse', 'HEAD'), text=True).strip(),
                  runtime={name: importlib.metadata.version(name)
                           for name in ('jax', 'jaxlib', 'libtpu')},
                  devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
                  scope='isolated byte adapter and grouping, no exchange or timing',
                  stages={}, exact=False)
    path = output / 'boundary.json'
    def save():
        path.write_text(json.dumps(report, indent=2))
    save()
    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    spec = jax.sharding.PartitionSpec('core', None, None)
    sharding = jax.sharding.NamedSharding(mesh, spec)
    wire, planes, ranks, valid = host_inputs()
    report['input_sha256'] = digest((wire, planes, ranks, valid))

    def adapter(x):
        return pallas_wire_to_planes(x[0])[None]
    call = jax.jit(jax.shard_map(adapter, mesh=mesh, in_specs=(spec,),
                                out_specs=spec, check_vma=False))
    arg = jax.device_put(wire, sharding)
    lowered = call.lower(arg)
    (output / 'adapter.mlir').write_text(lowered.as_text())
    exe = lowered.compile()
    (output / 'adapter.hlo.txt').write_text(exe.as_text())
    actual = np.asarray(jax.block_until_ready(exe(arg)))
    report['stages']['adapter'] = assess(actual, planes)
    save()

    def group(x, r, v):
        return pallas_group_final_records(x[0], r[0], v[0])[None]
    call = jax.jit(jax.shard_map(group, mesh=mesh, in_specs=(spec, spec, spec),
                                out_specs=spec, check_vma=False))
    args = tuple(jax.device_put(x, sharding) for x in (planes, ranks, valid))
    lowered = call.lower(*args)
    (output / 'group.mlir').write_text(lowered.as_text())
    exe = lowered.compile()
    (output / 'group.hlo.txt').write_text(exe.as_text())
    actual = np.asarray(jax.block_until_ready(exe(*args)))
    report['stages']['group_from_host_planes'] = assess(actual, host_grouped(planes, ranks, valid))
    report['exact'] = all(stage['exact'] for stage in report['stages'].values())
    save()
    if not report['exact']:
        raise RuntimeError('isolated response boundary mismatch; see boundary.json')


if __name__ == '__main__':
    main()
