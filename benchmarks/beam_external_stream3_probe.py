"""Eight-TPU composed S3 replay against saved original CPU C++ outputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_external_sort import pallas_external_stream3


def local_stream3(w,p,c,t,*,local_rank=None):
    result = pallas_external_stream3(w[0],p[0],c[0,0],t[0,0],
        local_rank=local_rank,world_size=8)
    return tuple(x[None] for x in result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    root = Path('tests/fixtures/external_stream3')
    manifest_bytes = (root/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        fixture_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        provenance=manifest,cases=[],scope='composed S3 before collector/RDMA; CPU source oracle, not CUDA')
    def save():
        (output/'external_stream3.json').write_text(json.dumps(report,indent=2))
    save()
    print(json.dumps(report),flush=True)
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec('core',None,None)
    sharding = jax.sharding.NamedSharding(mesh,p)
    for fixture in manifest['cases']:
        path = root/fixture['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != fixture['sha256']:
            raise ValueError('fixture hash mismatch')
        with np.load(path,allow_pickle=False) as data:
            inputs = tuple(data[k] for k in ('words','payload','counts','thresholds'))
            expected = tuple(data[f'expected_{i}'] for i in range(5))
        case = dict(capacity=fixture['capacity'],fixture_sha256=fixture['sha256'],
                    input_sha256=digest(inputs),expected_sha256=digest(expected))
        report['cases'].append(case)
        save()
        print(json.dumps(dict(event='compile_start',**case)),flush=True)
        args = tuple(jax.device_put(x,sharding) for x in inputs)
        fn = jax.jit(jax.shard_map(local_stream3,mesh=mesh,in_specs=(p,)*4,
                                   out_specs=(p,)*5,check_vma=False))
        start = time.perf_counter()
        exe = fn.lower(*args).compile()
        case['compile_seconds'] = time.perf_counter()-start
        (output/f"stream3_{fixture['capacity']}.hlo.txt").write_text(exe.as_text())
        actual = tuple(np.asarray(x) for x in jax.block_until_ready(exe(*args)))
        mismatch = sum(int(np.count_nonzero(a != e)) for a,e in zip(actual,expected,strict=True))
        case.update(mismatches=mismatch,exact=mismatch == 0,output_sha256=digest(actual))
        save()
        if mismatch:
            raise RuntimeError('composed S3 mismatch')
        for _ in range(3):
            jax.block_until_ready(exe(*args))
        samples = []
        for _ in range(21):
            start = time.perf_counter_ns()
            jax.block_until_ready(exe(*args))
            samples.append((time.perf_counter_ns()-start)/1e6)
        case['timing'] = dict(warmup=3,repeats=21,samples_ms=samples,
            median_ms=float(np.median(samples)),p10_ms=float(np.percentile(samples,10)),
            p90_ms=float(np.percentile(samples,90)))
        save()
        print(json.dumps(case),flush=True)


if __name__ == '__main__':
    main()
