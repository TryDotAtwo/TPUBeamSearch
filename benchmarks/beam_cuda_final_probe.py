"""Physical eight-TPU materialization versus immutable actual-CUDA outputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import jax
import numpy as np

from benchmarks.beam_cuda_final_fixture import load_cases
from tpu_beam_search.beam_final_materialize import pallas_materialize_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path('test_results/cuda_final_oracle_v2'))
    args = parser.parse_args()
    cases = load_cases(args.fixtures)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                  libtpu=importlib.metadata.version('libtpu'),
                  devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
                  scope='same CUDA fixtures on each TPU; materialization only, not exchange or full beam',
                  all_exact=False, cases=[])

    def save():
        (args.output / 'cuda_final.json').write_text(json.dumps(report, indent=2))

    save()
    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    p = jax.sharding.PartitionSpec
    specs = (p('core', None, None),) * 3 + (p('core', None),) * 2

    def local(*inputs):
        return tuple(x[None] for x in pallas_materialize_final(
            *(x[0] for x in inputs), state_len=120))

    fn = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=specs,
                             out_specs=(p('core', None, None),) * 2, check_vma=False))
    for case in cases:
        row = dict(name=case['name'], input_sha256=case['input_sha256'],
                   cuda_sha256=case['cuda_sha256'], exact=False)
        report['cases'].append(row)
        save()
        print(case['name'], flush=True)
        inputs = tuple(jax.device_put(np.broadcast_to(x, (8,) + x.shape).copy(),
                                      jax.sharding.NamedSharding(mesh, spec))
                       for x, spec in zip(case['inputs'], specs, strict=True))
        exe = fn.lower(*inputs).compile()
        (args.output / f"{case['name']}.hlo.txt").write_text(exe.as_text())
        wire, errors = (np.asarray(x) for x in jax.block_until_ready(exe(*inputs)))
        expected = np.broadcast_to(case['expected'], wire.shape)
        count = int(case['inputs'][3][0])
        hashes = [hashlib.sha256(x[:count].tobytes()).hexdigest() for x in wire]
        mismatches = int(np.count_nonzero(wire != expected))
        invalid = errors[:, 0, 0].tolist()
        exact = not mismatches and not any(invalid) and all(x == case['cuda_sha256'] for x in hashes)
        row.update(mismatches=mismatches, invalid_counts=invalid, device_sha256=hashes, exact=exact)
        save()
        if not exact:
            np.savez_compressed(args.output / f"{case['name']}_failure.npz", wire=wire, errors=errors, expected=expected)
            raise RuntimeError(f"{case['name']}: CUDA/TPU mismatch")
    report['all_exact'] = True
    save()


if __name__ == '__main__':
    main()
