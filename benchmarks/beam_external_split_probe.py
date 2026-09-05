"""Physical external split gate, supplied owners; not composed Stream3."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_external_sort import pallas_external_stream3_split


def local_split(w, o, c, *, local_rank=None):
    result = pallas_external_stream3_split(w[0], o[0], c[0],
        local_rank=local_rank, world_size=8)
    return tuple(x[None] for x in result)


def oracle(w, owners, count, rank):
    records = w.copy()
    records[7] = (rank << 16) | (owners << 8) | (w[7] & 255)
    local = [i for i in range(count) if owners[i] == rank]
    remote = sorted((i for i in range(count) if owners[i] != rank),
                    key=lambda i: (int(owners[i]), i))
    outputs = []
    for ids in (local, remote):
        out = np.zeros_like(w)
        out[6] = np.uint32(0xffffffff)
        out[:, :len(ids)] = records[:, ids]
        outputs.append(out)
    lc, counts, offsets = (np.zeros((1,128),np.uint32) for _ in range(3))
    lc[0,0] = len(local)
    for i in remote:
        counts[0,owners[i]] += 1
    offsets[0,1:9] = np.cumsum(counts[0,:8])
    return (*outputs, lc, counts, offsets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices], cases=[],
        scope='external split with supplied owners; no dedup/hash/RDMA/collector')
    def save():
        (output/'external_split.json').write_text(json.dumps(report,indent=2))
    save()
    print(json.dumps(report),flush=True)
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec('core',None,None)
    sharding = jax.sharding.NamedSharding(mesh,p)
    for n in (256,512,1024):
        rng = np.random.default_rng(20260905+n)
        words = rng.integers(0,2**32,(8,8,n),dtype=np.uint32)
        owners = rng.integers(0,8,(8,1,n),dtype=np.uint32)
        owners[1] = 1
        owners[2] = 7
        controls = np.zeros((8,1,128),np.uint32)
        controls[:,0,0] = [0,n,n,n-1,129,128,1,n]
        reference = tuple(np.stack(values) for values in zip(*[
            oracle(words[r],owners[r,0],int(controls[r,0,0]),r) for r in range(8)]))
        args = tuple(jax.device_put(x,sharding) for x in (words,owners,controls))
        case = dict(capacity=n,input_sha256=digest((words,owners,controls)))
        report['cases'].append(case)
        save()
        print(json.dumps(dict(event='compile_start',**case)),flush=True)
        fn = jax.jit(jax.shard_map(local_split,mesh=mesh,in_specs=(p,p,p),
                                   out_specs=(p,)*5,check_vma=False))
        start = time.perf_counter()
        exe = fn.lower(*args).compile()
        case['compile_seconds'] = time.perf_counter()-start
        (output/f'split_{n}.hlo.txt').write_text(exe.as_text())
        actual = tuple(np.asarray(x) for x in jax.block_until_ready(exe(*args)))
        mismatches = sum(int(np.count_nonzero(a != b)) for a,b in zip(actual,reference))
        case.update(mismatches=mismatches,exact=mismatches == 0,
                    output_sha256=digest(actual),expected_sha256=digest(reference))
        save()
        if mismatches:
            raise RuntimeError('external split mismatch')
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
