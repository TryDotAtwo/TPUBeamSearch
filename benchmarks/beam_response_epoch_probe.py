"""Eight physical TPU response routing gate; no frontier publication/timing."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from benchmarks.beam_response_epoch_fixture import fixtures,expected_epoch
from tpu_beam_search.beam_final_response_routing import pallas_prepare_final_response_exchange
from tpu_beam_search.beam_final_response_chunk import make_final_response_chunk_call


def local_calls(mesh):
    def prepare(wire,requests,control,validation):
        return tuple(x[None] for x in pallas_prepare_final_response_exchange(
            wire[0],requests[0],control[0],validation[0],world_size=mesh.size))
    call=make_final_response_chunk_call(mesh,wire_width=128)
    def epoch(grouped,intervals,error,index):
        return tuple(x[None] for x in call(grouped[0],intervals[0],error[0],index))
    return prepare,epoch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output
    output.mkdir(parents=True,exist_ok=False)
    devices=jax.devices()
    if len(devices)!=8 or any(d.platform!='tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime={k:importlib.metadata.version(k) for k in ('jax','jaxlib','libtpu')},
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope='routed response epochs, no materialization/publication or timing',
        exact=False,cases=[])
    def save():
        (output/'response_epoch.json').write_text(json.dumps(report,indent=2))
    save()
    mesh=jax.sharding.Mesh(np.asarray(devices),('core',))
    spec=jax.sharding.PartitionSpec('core',None,None)
    replicated=jax.sharding.PartitionSpec()
    sharding=jax.sharding.NamedSharding(mesh,spec)
    prepare,epoch=local_calls(mesh)
    prep=jax.jit(jax.shard_map(prepare,mesh=mesh,in_specs=(spec,)*4,
        out_specs=(spec,)*3,check_vma=False))
    step=jax.jit(jax.shard_map(epoch,mesh=mesh,in_specs=(spec,spec,spec,replicated),
        out_specs=(spec,spec),check_vma=False))
    prep_exe=step_exe=None
    for name,inputs in fixtures():
        row=dict(name=name,input_sha256=digest(inputs),epochs=[],exact=False)
        report['cases'].append(row)
        save()
        args=tuple(jax.device_put(x,sharding) for x in inputs)
        if prep_exe is None:
            lowered=prep.lower(*args)
            (output/'prepare.mlir').write_text(lowered.as_text())
            prep_exe=lowered.compile()
            (output/'prepare.hlo.txt').write_text(prep_exe.as_text())
        prepared=jax.block_until_ready(prep_exe(*args))
        for index in range(3):
            wire,requests,control,validation=inputs
            expected=expected_epoch(wire,requests,control[:,0,0],
                (control[:,1,0]!=0)|(validation[:,0,0]!=0),index)
            item=dict(epoch=index,expected_sha256=digest(expected),exact=False)
            row['epochs'].append(item)
            save()
            print(name,index,flush=True)
            number=jax.device_put(np.array([index],np.uint32),
                jax.sharding.NamedSharding(mesh,replicated))
            if step_exe is None:
                lowered=step.lower(*prepared,number)
                (output/'epoch.mlir').write_text(lowered.as_text())
                step_exe=lowered.compile()
                (output/'epoch.hlo.txt').write_text(step_exe.as_text())
            actual=tuple(np.asarray(x) for x in jax.block_until_ready(step_exe(*prepared,number)))
            mismatch=[[int(np.count_nonzero(a[r]!=e[r])) for r in range(8)]
                      for a,e in zip(actual,expected,strict=True)]
            exact=not any(any(x) for x in mismatch)
            item.update(exact=exact,mismatches=mismatch,output_sha256=digest(actual))
            save()
            if not exact:
                raise RuntimeError(f'{name} epoch{index}: response mismatch')
        row['exact']=True
        save()
    report['exact']=True
    save()


if __name__=='__main__':
    main()
