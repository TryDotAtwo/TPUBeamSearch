"""Eight-TPU functional hash collector gate; no resident/overlap claim."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_collector import pallas_collect, reserve_group


def local_collect(*args):
    return tuple(x[None] for x in pallas_collect(*(a[0] for a in args)))


def reference_shard(words,shards):
    mask = (1 << 64)-1
    def mix(x):
        x = ((x^(x >> 30))*0xbf58476d1ce4e5b9)&mask
        x = ((x^(x >> 27))*0x94d049bb133111eb)&mask
        return x^(x >> 31)
    lo = int(words[0]) | (int(words[1]) << 32)
    hi = int(words[2]) | (int(words[3]) << 32)
    return mix(lo^(((hi << 32)|(hi >> 32))&mask)^0x13198a2e03707344^
               mix((hi+0x9e3779b97f4a7c15)&mask))%shards


def fixtures():
    rng = np.random.default_rng(291)
    a = rng.integers(0,2**32,(8,3,8,128),dtype=np.uint32)
    b = rng.integers(0,2**32,a.shape,dtype=np.uint32)
    words = rng.integers(0,2**32,(8,8,256),dtype=np.uint32)
    controls = np.zeros((8,3,8,128),np.uint32)
    count = np.zeros((8,1,128),np.uint32)
    count[:,0,0] = [213,0,127,256,1,213,213,213]
    controls[1,:,4:6,0] = 1
    controls[2,:,0,0] = 100
    controls[3,:,0:2,0] = 128
    controls[4,:,4,0] = 1
    controls[5,:,7,0] = 1
    controls[6,:,0:2,0] = 64
    controls[7,:,6,0] = 1
    expected = [a.copy(),b.copy(),controls.copy(),np.zeros((8,1,128),np.uint32)]
    for r in range(8):
        ids = [reference_shard(words[r,:,i],3) for i in range(int(count[r,0,0]))]
        groups = [words[r][:,np.flatnonzero(np.asarray(ids) == s)] for s in range(3)]
        plans = [reserve_group(capacity=128,
            clean=tuple(int(x) for x in controls[r,s,:2,0]),dirty=(0,0),
            processing=tuple(int(x) for x in controls[r,s,4:6,0]),
            current=int(controls[r,s,6,0]),amount=g.shape[1]) for s,g in enumerate(groups)]
        failed = bool(np.any(controls[r,:,7,0])) or any(p.fatal_overflow for p in plans)
        expected[3][r,0,0] = failed
        expected[2][r,:,7,0] = failed
        if failed:
            continue
        for s,(p,g) in enumerate(zip(plans,groups)):
            if p.buffer is not None:
                expected[p.buffer][r,s,:,p.offset:p.offset+g.shape[1]] = g
                expected[2][r,s,2:4,0] = p.dirty
                expected[2][r,s,6,0] = p.buffer
    return (a,b,words,controls,count),tuple(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    inputs,expected = fixtures()
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope='functional hash collector, three shards/rank; not resident, RDMA or full beam',
        input_sha256=digest(inputs),expected_sha256=digest(expected),exact=False)
    def save():
        (output/'collector_full.json').write_text(json.dumps(report,indent=2))
    save()
    print(json.dumps(report),flush=True)
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    specs = tuple(jax.sharding.PartitionSpec('core',*(None for _ in x.shape[1:])) for x in inputs)
    out_specs = tuple(jax.sharding.PartitionSpec('core',*(None for _ in x.shape[1:])) for x in expected)
    args = tuple(jax.device_put(x,jax.sharding.NamedSharding(mesh,p)) for x,p in zip(inputs,specs))
    fn = jax.jit(jax.shard_map(local_collect,mesh=mesh,in_specs=specs,out_specs=out_specs,check_vma=False))
    start = time.perf_counter()
    exe = fn.lower(*args).compile()
    report['compile_seconds'] = time.perf_counter()-start
    (output/'collector_full.hlo.txt').write_text(exe.as_text())
    actual = tuple(np.asarray(x) for x in jax.block_until_ready(exe(*args)))
    mismatch = [int(np.count_nonzero(x != y)) for x,y in zip(actual,expected,strict=True)]
    report.update(exact=not any(mismatch),mismatches=mismatch,output_sha256=digest(actual))
    save()
    if any(mismatch):
        raise RuntimeError('full collector mismatch')
    for _ in range(3):
        jax.block_until_ready(exe(*args))
    samples = []
    for _ in range(21):
        start = time.perf_counter_ns()
        jax.block_until_ready(exe(*args))
        samples.append((time.perf_counter_ns()-start)/1e6)
    report['timing'] = dict(warmup=3,repeats=21,samples_ms=samples,
        median_ms=float(np.median(samples)),p10_ms=float(np.percentile(samples,10)),
        p90_ms=float(np.percentile(samples,90)))
    save()
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
