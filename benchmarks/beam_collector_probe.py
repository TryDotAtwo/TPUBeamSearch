"""Serialized functional collector gate, not concurrent publication."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_collector import pallas_collector_append, pallas_collector_append_group, reserve_group


def local_append(*args):
    return tuple(x[None] for x in pallas_collector_append(*(a[0] for a in args)))


def local_append_group(*args):
    return tuple(x[None] for x in pallas_collector_append_group(*(a[0] for a in args)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',action='store_true')
    options = parser.parse_args()
    output = options.output
    capacity = 512 if options.group else 256
    incoming_capacity = 512 if options.group else 128
    output.mkdir(parents=True,exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],cases=[],
        scope='serialized functional one-shard collector; no concurrency/alias/overlap')
    def save():
        (output/'collector.json').write_text(json.dumps(report,indent=2))
    save()
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec('core',None,None)
    sharding = jax.sharding.NamedSharding(mesh,p)
    rng = np.random.default_rng(7823)
    a = rng.integers(0,2**32,(8,8,capacity),dtype=np.uint32)
    b = rng.integers(0,2**32,a.shape,dtype=np.uint32)
    incoming = rng.integers(0,2**32,(8,8,incoming_capacity),dtype=np.uint32)
    control = np.zeros((8,8,128),np.uint32)
    count = np.zeros((8,1,128),np.uint32)
    scenarios = [((129,0),(0,0),0,127),((250,0),(0,0),0,128),
                 ((0,129),(1,0),0,127),((250,250),(0,0),0,128),
                 ((256,256),(1,1),0,0),((0,0),(0,0),1,128),
                 ((5,6),(0,1),1,1),((10,20),(0,0),0,1)]
    if options.group:
        scenarios = [((300,0),(0,0),0,256),((300,300),(0,0),0,256),
                     ((255,0),(0,0),0,257),((0,255),(1,0),0,257),
                     ((512,512),(1,1),0,0),((0,0),(0,0),1,512),
                     ((5,6),(0,1),1,385),((10,20),(0,0),0,256)]
    expected = [a.copy(),b.copy(),control.copy()]
    for r,(used,busy,current,amount) in enumerate(scenarios):
        control[r,:2,0] = used
        control[r,4:6,0] = busy
        control[r,6,0] = current
        control[r,7,0] = int(r == 7)
        count[r,0,0] = amount
        expected[2][r] = control[r]
        if r == 7:
            continue
        reservation = reserve_group(capacity=capacity,clean=used,dirty=(0,0),
            processing=busy,current=current,amount=amount)
        if reservation.buffer is not None:
            target = reservation.buffer
            expected[target][r,:,reservation.offset:reservation.offset+amount] = incoming[r,:,:amount]
            expected[2][r,2:4,0] = reservation.dirty
            expected[2][r,6,0] = target
        expected[2][r,7,0] = reservation.fatal_overflow
    inputs = (a,b,incoming,control,count)
    case = dict(group=options.group,capacity=capacity,incoming_capacity=incoming_capacity,
                input_sha256=digest(inputs),expected_sha256=digest(expected))
    report['cases'].append(case)
    save()
    print(json.dumps(report),flush=True)
    args = tuple(jax.device_put(x,sharding) for x in inputs)
    adapter = local_append_group if options.group else local_append
    fn = jax.jit(jax.shard_map(adapter,mesh=mesh,in_specs=(p,)*5,out_specs=(p,)*3,check_vma=False))
    start = time.perf_counter()
    exe = fn.lower(*args).compile()
    case['compile_seconds'] = time.perf_counter()-start
    (output/'collector.hlo.txt').write_text(exe.as_text())
    actual = tuple(np.asarray(x) for x in jax.block_until_ready(exe(*args)))
    mismatch = sum(int(np.count_nonzero(x != y)) for x,y in zip(actual,expected,strict=True))
    case.update(exact=mismatch == 0,mismatches=mismatch,output_sha256=digest(actual))
    save()
    if mismatch:
        raise RuntimeError('collector mismatch')
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
