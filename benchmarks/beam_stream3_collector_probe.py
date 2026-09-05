"""Physical eight-TPU S3/exchange/collector gate against saved CPU fixtures."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.build_external_stream3_fixtures import sha
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_remote_exchange import make_stream3_collect_call


def make_local_program(mesh):
    call = make_stream3_collect_call(mesh)
    def local(*args):
        return tuple(x[None] for x in call(*(a[0] for a in args)))
    return local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    fixture_root = Path(__file__).resolve().parents[1]/'tests/fixtures/stream3_collector'
    manifest = json.loads((fixture_root/'manifest.json').read_text())
    fixture = fixture_root/manifest['file']
    if sha(fixture) != manifest['sha256']:
        raise RuntimeError('fixture SHA mismatch')
    with np.load(fixture,allow_pickle=False) as data:
        inputs = tuple(data[name] for name in ('a','b','controls','words','payload','counts','thresholds','neutral'))
        expected = tuple(data[f'expected_{i}'] for i in range(4))
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope='bounded128 S3/snapshot RDMA/functional collector; mixed expected fatal; not full beam or CUDA parity',
        reference=manifest,input_sha256=digest(inputs),expected_sha256=digest(expected),exact=False)
    def save():
        (output/'stream3_collector.json').write_text(json.dumps(report,indent=2))
    save()
    print(json.dumps(report),flush=True)
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    specs = tuple(jax.sharding.PartitionSpec('core',*(None for _ in x.shape[1:])) for x in inputs)
    out_specs = tuple(jax.sharding.PartitionSpec('core',*(None for _ in x.shape[1:])) for x in expected)
    args = tuple(jax.device_put(x,jax.sharding.NamedSharding(mesh,p)) for x,p in zip(inputs,specs))
    fn = jax.jit(jax.shard_map(make_local_program(mesh),mesh=mesh,
        in_specs=specs,out_specs=out_specs,check_vma=False))
    start = time.perf_counter()
    exe = fn.lower(*args).compile()
    report['compile_seconds'] = time.perf_counter()-start
    (output/'stream3_collector.hlo.txt').write_text(exe.as_text())
    actual = tuple(np.asarray(x) for x in jax.block_until_ready(exe(*args)))
    mismatch = [int(np.count_nonzero(x != y)) for x,y in zip(actual,expected,strict=True)]
    report.update(exact=not any(mismatch),mismatches=mismatch,output_sha256=digest(actual),
                  actual_fatal=actual[3][:,0,0].tolist())
    save()
    if any(mismatch):
        raise RuntimeError('S3/exchange/collector mismatch')
    for _ in range(3):
        jax.block_until_ready(exe(*args))
    samples = []
    for _ in range(21):
        start = time.perf_counter_ns()
        jax.block_until_ready(exe(*args))
        samples.append((time.perf_counter_ns()-start)/1e6)
    report['timing'] = dict(scope='diagnostic mixed-success/fatal fixture, not beam throughput',
        warmup=3,repeats=21,samples_ms=samples,median_ms=float(np.median(samples)),
        p10_ms=float(np.percentile(samples,10)),p90_ms=float(np.percentile(samples,90)))
    save()
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
