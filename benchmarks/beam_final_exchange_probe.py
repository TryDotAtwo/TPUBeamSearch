"""Eight-TPU final chunk snapshots; correctness gate, not full beam timing."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_final_exchange import make_final_chunk_exchange


def fixtures():
    cases=[]
    for repeat in range(2):
        for name in ('empty','self','one_to_all','all_to_one','uneven','full','bad_count','error'):
            controls=np.zeros((8,8,2,128),np.uint32)
            payload=np.zeros((8,8,4,128),np.uint32)
            for source in range(8):
                for destination in range(8):
                    count=(0 if name=='empty' else (source+1 if source==destination else 0)
                        if name=='self' else (128 if source==3 else 0)
                        if name=='one_to_all' else (source+1 if destination==5 else 0)
                        if name=='all_to_one' else 128 if name=='full'
                        else (source*17+destination*31+repeat)%129)
                    controls[source,destination,0,0]=count
                    values=np.arange(4*128,dtype=np.uint32).reshape(4,128)
                    payload[source,destination,:,:count]=values[:,:count]+np.uint32(1+source*100000+destination*1000+repeat*1000000)
            if name=='bad_count':
                controls[6,2,0,0]=129
            if name=='error':
                controls[2,0,1,0]=1
            error=np.zeros((8,1,128),np.uint32)
            if name in ('bad_count','error'):
                expected=np.zeros_like(payload)
                counts=np.zeros((8,8,1,128),np.uint32)
                error[:,0,0]=1
            else:
                expected=payload.transpose(1,0,2,3).copy()
                counts=controls[:,:,:1,:].transpose(1,0,2,3).copy()
            cases.append((f'round{repeat}_{name}',(payload,controls),(expected,counts,error)))
    return cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    devices=jax.devices()
    if len(devices)!=8 or any(d.platform!='tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope='repeated final chunk exchange snapshots, not full final or beam; no timing claim',exact=False,cases=[])
    def save():
        (output/'final_exchange.json').write_text(json.dumps(report,indent=2))
    save()
    mesh=jax.sharding.Mesh(np.asarray(devices),('core',))
    spec=jax.sharding.PartitionSpec('core',None,None,None)
    error_spec=jax.sharding.PartitionSpec('core',None,None)
    sharding=jax.sharding.NamedSharding(mesh,spec)
    call=make_final_chunk_exchange(mesh,planes=4)
    def local(payload,controls):
        return tuple(x[None] for x in call(payload[0],controls[0]))
    fn=jax.jit(jax.shard_map(local,mesh=mesh,in_specs=(spec,spec),
        out_specs=(spec,spec,error_spec),check_vma=False))
    cases=fixtures()
    args=tuple(jax.device_put(x,sharding) for x in cases[0][1])
    start=time.perf_counter()
    exe=fn.lower(*args).compile()
    report['compile_seconds']=time.perf_counter()-start
    (output/'final_exchange.hlo.txt').write_text(exe.as_text())
    save()
    for name,inputs,expected in cases:
        row=dict(name=name,input_sha256=digest(inputs),expected_sha256=digest(expected),exact=False)
        report['cases'].append(row)
        save()
        print(name,flush=True)
        actual=tuple(np.asarray(x) for x in jax.block_until_ready(exe(*(jax.device_put(x,sharding) for x in inputs))))
        mismatch=[int(np.count_nonzero(x!=y)) for x,y in zip(actual,expected,strict=True)]
        row.update(mismatches=mismatch,output_sha256=digest(actual),exact=not any(mismatch))
        if any(mismatch):
            path=output/f'{name}_failure.npz'
            np.savez_compressed(path,**{f'input{i}':x for i,x in enumerate(inputs)},
                **{f'expected{i}':x for i,x in enumerate(expected)},**{f'actual{i}':x for i,x in enumerate(actual)})
            row['failure_arrays']=path.name
        save()
        if any(mismatch):
            raise RuntimeError(f'{name}: final exchange mismatch')
    report['exact']=True
    save()


if __name__=='__main__':
    main()
