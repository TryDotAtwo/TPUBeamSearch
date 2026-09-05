"""Eight-device reserved S4 job gate, not scheduler or full beam throughput."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_s4_commit import pallas_run_reserved_s4


def fixtures():
    words = np.zeros((8,8,128),np.uint32)
    a,b = np.full((8,1,128),11,np.uint32),np.full((8,1,128),22,np.uint32)
    control = np.zeros((8,4,128),np.uint32)
    threshold = np.zeros((8,1),np.uint32)
    expected = [words.copy(),a.copy(),b.copy(),control.copy()]
    expected[0][:,6,:] = np.uint32(0xffffffff)
    for rank in range(8):
        words[rank,0,:4] = [1,2,1,3]
        words[rank,6,:4] = [7,5,3,8]
        words[rank,4,:4] = [10,20,30,40]
        # Both active slots, empty results and nonempty results on each slot.
        active = (rank//2)%2
        threshold[rank,0] = 7 if rank%2 == 0 else 0
        control[rank,:,0] = (2,2,1,active)
        hist = np.zeros((1,128),np.uint32)
        count = 0
        if rank%2 == 0:
            expected[0][rank,:,:2] = words[rank][:,[2,1]]
            hist[0,3],hist[0,5] = 1,1
            count = 2
        expected[2 if active == 0 else 1][rank] = hist
        expected[3][rank,:,0] = (count,0,0,active^1)
    return (words,a,b,control,threshold),tuple(expected)


def local_job(*args):
    return tuple(x[None] for x in pallas_run_reserved_s4(*(a[0] for a in args),bins=128))


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
        scope='reserved128 S4 dedup/histogram/commit; no ready queue, S5, production histogram width or beam throughput',
        input_sha256=digest(inputs),expected_sha256=digest(expected),exact=False)
    def save():
        (output/'s4_reserved.json').write_text(json.dumps(report,indent=2))
    save()
    print(json.dumps(report),flush=True)
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    specs = tuple(jax.sharding.PartitionSpec('core',*(None for _ in x.shape[1:])) for x in inputs)
    out_specs = tuple(jax.sharding.PartitionSpec('core',*(None for _ in x.shape[1:])) for x in expected)
    args = tuple(jax.device_put(x,jax.sharding.NamedSharding(mesh,p)) for x,p in zip(inputs,specs))
    fn = jax.jit(jax.shard_map(local_job,mesh=mesh,in_specs=specs,out_specs=out_specs,check_vma=False))
    start = time.perf_counter()
    exe = fn.lower(*args).compile()
    report['compile_seconds'] = time.perf_counter()-start
    (output/'s4_reserved.hlo.txt').write_text(exe.as_text())
    actual = tuple(np.asarray(x) for x in jax.block_until_ready(exe(*args)))
    mismatch = [int(np.count_nonzero(x != y)) for x,y in zip(actual,expected,strict=True)]
    report.update(exact=not any(mismatch),mismatches=mismatch,output_sha256=digest(actual))
    save()
    if any(mismatch):
        raise RuntimeError('reserved S4 mismatch')
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
