"""Repeated eight-TPU request epochs; no histogram or throughput claim."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_s5_request import make_s5_request_call
from tpu_beam_search.beam_s5_histogram_exchange import make_s5_histogram_call
from tpu_beam_search.beam_histogram_pair_sum import pallas_sum_histogram_pairs


def fixtures():
    cases = []
    for repeat in range(2):
        for phase in range(10):
            request = np.zeros((8,1,128),np.uint32)
            if phase == 9:
                request[:,0,0] = 1
            elif phase:
                request[phase-1,0,0] = 1
            expected = np.zeros_like(request)
            expected[:,0,0] = int(phase != 0)
            cases.append((f'round{repeat}_case{phase}',request,expected))
    return cases


def histogram_fixtures():
    rng = np.random.default_rng(603)
    mixed = rng.integers(0,1<<59,(8,256),dtype=np.uint64)
    mixed[:,0] = 0xffffffff
    mixed[:,129] = 0x100000001
    single = np.zeros_like(mixed)
    single[5] = np.arange(256,dtype=np.uint64)+(np.uint64(9)<<np.uint64(32))
    cases = []
    for repeat in range(2):
        for phase,values in enumerate((np.zeros_like(mixed),mixed,single)):
            packed = np.stack((values.astype(np.uint32),
                (values>>np.uint64(32)).astype(np.uint32)),axis=1)
            total = values.sum(axis=0,dtype=np.uint64)
            expected = np.stack((total.astype(np.uint32),
                (total>>np.uint64(32)).astype(np.uint32)))
            cases.append((f'round{repeat}_hist{phase}',packed,
                          np.broadcast_to(expected,(8,2,256)).copy()))
    return cases


def recovery_fixtures(kind):
    base = histogram_fixtures()
    cases = []
    for repeat in range(2):
        for step,index in enumerate((1,0,1,2)):
            _,source,total = base[index]
            wire = np.stack([np.concatenate([source[(rank-offset)%8] for offset in range(8)],axis=0)
                             for rank in range(8)])
            request,expected = (source,wire) if kind == 'wire' else (wire,total) if kind == 'reduction' else (source,total)
            if kind == 'own':
                request,expected = source,source.copy()
            cases.append((f'recovery{repeat}_{step}',request,expected))
    return cases


def save_failure_arrays(output,name,request,expected,actual):
    path = Path(output)/f'{name}_failure.npz'
    np.savez_compressed(path,input=request,expected=expected,actual=actual)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--kind',choices=('request','histogram','wire','reduction','combined','own'),default='request')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope=f'{args.kind} only; repeated same executable; no full S5/ownership or timing claim',
        cases=[],exact=False)
    def save():
        (output/f's5_{args.kind}.json').write_text(json.dumps(report,indent=2))
    save()
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    spec = jax.sharding.PartitionSpec('core',None,None)
    sharding = jax.sharding.NamedSharding(mesh,spec)
    call = (make_s5_request_call(mesh) if args.kind == 'request'
            else pallas_sum_histogram_pairs if args.kind == 'reduction'
            else make_s5_histogram_call(mesh,width=256,return_wire=args.kind == 'wire',own_only=args.kind == 'own'))
    def local(request):
        return call(request[0])[None]
    fn = jax.jit(jax.shard_map(local,mesh=mesh,in_specs=spec,out_specs=spec,check_vma=False))
    cases = (fixtures() if args.kind == 'request' else histogram_fixtures()
             if args.kind == 'histogram' else recovery_fixtures(args.kind))
    argument = jax.device_put(cases[0][1],sharding)
    start = time.perf_counter()
    exe = fn.lower(argument).compile()
    report['compile_seconds'] = time.perf_counter()-start
    (output/f's5_{args.kind}.hlo.txt').write_text(exe.as_text())
    save()
    for name,request,expected in cases:
        row = dict(name=name,input_sha256=digest((request,)),expected_sha256=digest((expected,)),exact=False)
        report['cases'].append(row)
        save()
        print(name,flush=True)
        actual = np.asarray(jax.block_until_ready(exe(jax.device_put(request,sharding))))
        mismatches = int(np.count_nonzero(actual != expected))
        row.update(mismatches=mismatches,exact=mismatches == 0,output_sha256=digest((actual,)))
        if mismatches:
            row['failure_arrays'] = save_failure_arrays(output,name,request,expected,actual).name
        save()
        if mismatches:
            raise RuntimeError(f'{name}: {args.kind} mismatch')
    report['exact'] = True
    save()
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
