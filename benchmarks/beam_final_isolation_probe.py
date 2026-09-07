"""Extracted materialization stages with an eight-TPU diagnostic runner."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tpu_beam_search.beam_final_materialize import pallas_materialize_final
from .beam_final_isolation_bundle import MODES


def report_exact(rows):
    return (len(rows)==2 and {r.get('count') for r in rows}=={0,2}
            and all(r.get('mismatches')==[0]*8 for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',False)
    devices = jax.devices()
    if len(devices)!=8 or any(d.platform!='tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    args.output.mkdir(parents=True,exist_ok=True)
    report = dict(mode=args.mode,exact=False,cases=[],
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope='isolated operation bytes, not full final/beam or performance')
    def save():
        (args.output/'probe.json').write_text(json.dumps(report,indent=2))
    save()
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec
    specs = (p('core',None,None),)*3+(p('core',None),)
    def local(*xs):
        return run_probe(args.mode,*(x[0] for x in xs))[None]
    fn = jax.jit(jax.shard_map(local,mesh=mesh,in_specs=specs,
                              out_specs=p('core',None,None),check_vma=False))
    for count in (0,2):
        inputs,expected = fixture(args.mode,count)
        row = dict(count=count,status='pending',
            input_sha256=hashlib.sha256(b''.join(x.tobytes() for x in inputs)).hexdigest(),
            expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
        report['cases'].append(row)
        save()
        xs = tuple(jax.device_put(np.broadcast_to(x,(8,)+x.shape).copy(),
                   jax.sharding.NamedSharding(mesh,spec)) for x,spec in zip(inputs,specs))
        lowered = fn.lower(*xs)
        (args.output/f'count{count}.lowered.mlir').write_text(lowered.as_text())
        row['status']='compiling'
        save()
        exe = lowered.compile()
        (args.output/f'count{count}.hlo.txt').write_text(exe.as_text())
        actual = np.asarray(jax.block_until_ready(exe(*xs)))
        row.update(status='executed',mismatches=[int(np.count_nonzero(x!=expected)) for x in actual],
                   output_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in actual])
        save()
    report['exact']=report_exact(report['cases'])
    save()
    raise SystemExit(0 if report['exact'] else 1)


def fixture(mode,count):
    if mode not in MODES or count not in (0,2):
        raise ValueError('invalid probe fixture')
    parents = (np.arange(7*128).reshape(7,128)%256).astype(np.uint8)
    requests = np.zeros((4,128),np.uint32)
    requests[0,:2] = [6,0]
    requests[2,:2] = [7,5]
    perm = np.arange(127,-1,-1,dtype=np.int32)[None,:]
    expected = np.zeros((128,128),np.uint8)
    for i,parent in enumerate((6,0)[:count]):
        child = parents[parent].copy()
        if mode not in ('dma','cast'):
            child = child[::-1].copy()
        if mode in ('packing','production'):
            child[120:] = 0
            child[120:124] = np.frombuffer(int(requests[2,i]).to_bytes(4,'little'),np.uint8)
        expected[i] = child
    return (parents,perm,requests,np.array([count],np.uint32)),expected


def run_probe(mode,parents,perm,requests,count,*,interpret=False):
    if mode not in MODES:
        raise ValueError('invalid probe mode')
    parents,perm,requests,count = map(jnp.asarray,(parents,perm,requests,count))
    if mode == 'production':
        return pallas_materialize_final(parents,perm,requests,count,
            jnp.array([8],jnp.uint32),state_len=120,interpret=interpret)[0]
    def kernel(p,g,r,c,out,stage,sem):
        i = pl.program_id(0)
        stage[...] = jnp.zeros((1,1,128),jnp.uint8)
        @pl.when(i.astype(jnp.uint32)<c[0])
        def valid():
            lanes = jnp.arange(128,dtype=jnp.int32)
            parent = jnp.sum(jnp.where(lanes==i,r[0],jnp.uint32(0)).astype(jnp.int32))
            load = pltpu.make_async_copy(p.at[pl.ds(parent,1),:,:],stage,sem)
            load.start()
            load.wait()
            if mode != 'dma':
                value = stage[0,0].astype(jnp.int32)
                if mode in ('gather_1d','packing'):
                    value = jnp.take_along_axis(value,g[0],axis=0,mode='promise_in_bounds')
                elif mode == 'gather_2d':
                    value = jnp.take_along_axis(value[None,:],g[...],axis=1,
                                               mode='promise_in_bounds')[0]
                elif mode == 'select_reduce':
                    # Diagnostic O(width squared) permutation, not a speed claim.
                    # Exactly one input lane contributes to each output lane.
                    value = jnp.sum(jnp.where(lanes[:,None]==g[...],
                                              value[:,None],jnp.int32(0)),axis=0)
                child = value.astype(jnp.uint8)
                if mode == 'packing':
                    target = jnp.sum(jnp.where(lanes==i,r[2],jnp.uint32(0)).astype(jnp.int32)).astype(jnp.uint32)
                    child = jnp.where(lanes<120,child,jnp.uint8(0))
                    for byte in range(4):
                        child = jnp.where(lanes==120+byte,
                            ((target>>jnp.uint32(byte*8))&jnp.uint32(255)).astype(jnp.uint8),child)
                stage[...] = child[None,None,:]
        store = pltpu.make_async_copy(stage,out.at[pl.ds(i,1),:,:],sem)
        store.start()
        store.wait()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((128,1,128),jnp.uint8),
        in_specs=(hbm,pl.BlockSpec((1,128)),pl.BlockSpec((4,128)),pl.BlockSpec((1,))),
        out_specs=hbm,grid=(128,),
        scratch_shapes=(pltpu.VMEM((1,1,128),jnp.uint8),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='isolate_final_'+mode)(parents[:,None,:],perm,requests,count)[:,0,:]


if __name__ == '__main__':
    main()
