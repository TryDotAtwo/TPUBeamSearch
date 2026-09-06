"""Physical eight-rank coverage agreement; not full final publication."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_final_agreement import make_final_coverage_agreement


def fixtures():
    for name in ('empty','uneven','duplicate','missing','extra','overflow','prior_error'):
        targets = np.full((8,1,256),0xffffffff,np.uint32)
        valid = np.zeros_like(targets)
        counts = np.zeros((8,1),np.uint32)
        prior = np.zeros((8,1,128),np.uint32)
        for rank in range(8):
            n = 0 if name == 'empty' else (0,1,127,128,129,17,2,256)[rank]
            counts[rank,0] = n
            slots = np.random.default_rng(615+rank).permutation(256)[:n]
            targets[rank,0,slots] = np.arange(n,dtype=np.uint32)
            valid[rank,0,slots] = 1
        if name == 'duplicate':
            slots = np.flatnonzero(valid[4,0])
            targets[4,0,slots[1]] = targets[4,0,slots[0]]
        if name == 'missing': valid[4,0,np.flatnonzero(valid[4,0])[0]] = 0
        if name == 'extra': valid[0,0,0] = 1
        if name == 'overflow': counts[4,0] = 257
        if name == 'prior_error': prior[6,0,0] = 0x80000000
        yield name,(targets,valid,counts,prior),int(name not in ('empty','uneven'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],exact=False,cases=[],
        scope='coverage common-error only; not frontier publication, DMA drain or full beam')
    def save(): (output/'final_coverage.json').write_text(json.dumps(report,indent=2))
    save()
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec
    specs = (p('core',None,None),p('core',None,None),p('core',None),p('core',None,None))
    call = make_final_coverage_agreement(mesh)
    def local(t,v,c,e): return tuple(x[None] for x in call(t[0],v[0],c[0],e[0]))
    fn = jax.jit(jax.shard_map(local,mesh=mesh,in_specs=specs,
        out_specs=(p('core',None,None),)*2,check_vma=False))
    exe = None
    for name,inputs,error in fixtures():
        row = dict(name=name,input_sha256=digest(inputs),expected_common_error=error,exact=False)
        report['cases'].append(row)
        save()
        args = tuple(jax.device_put(x,jax.sharding.NamedSharding(mesh,s)) for x,s in zip(inputs,specs))
        if exe is None:
            exe = fn.lower(*args).compile()
            (output/'final_coverage.hlo.txt').write_text(exe.as_text())
        common,summary = (np.asarray(x) for x in jax.block_until_ready(exe(*args)))
        expected = np.zeros((8,1,128),np.uint32)
        expected[:,0,0] = error
        local_bad = (summary[:,0,0] != 0)
        expected_bad = np.zeros(8,bool)
        if error and name != 'prior_error': expected_bad[0 if name == 'extra' else 4] = True
        exact = bool(np.array_equal(common,expected) and np.array_equal(local_bad,expected_bad))
        row.update(exact=exact,output_sha256=digest((common,summary)),invalid_counts=summary[:,0,0].tolist())
        save()
        if not exact:
            np.savez_compressed(output/f'{name}_failure.npz',common=common,summary=summary)
            raise RuntimeError(f'{name}: coverage mismatch')
    report['exact'] = True
    save()


if __name__ == '__main__': main()
