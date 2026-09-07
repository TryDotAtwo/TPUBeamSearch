"""Eight-device scatter/chain diagnostic; not a frontier publication caller."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from .beam_final_scatter_fixture import make_case
from tpu_beam_search.beam_final_materialize import pallas_materialize_final
from tpu_beam_search.beam_final_scatter import pallas_scatter_final_responses


def run_case(mode,parents,perm,requests,count,frontier,wire,*,interpret=False):
    parents,perm,requests,count,frontier,wire = map(jnp.asarray,
        (parents,perm,requests,count,frontier,wire))
    if mode == 'integrated':
        wire,validation = pallas_materialize_final(parents,perm,requests,count,
            jnp.array([frontier.shape[0]],jnp.uint32),state_len=120,interpret=interpret)
        # This diagnostic only chains valid fixtures. Never publish its output.
    elif mode != 'scatter':
        raise ValueError('unknown scatter probe mode')
    result,errors = pallas_scatter_final_responses(frontier,wire,count,
        state_len=120,interpret=interpret)
    if mode == 'integrated':
        errors = errors.at[0,0].add(validation[0,0])
    return result,errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('scatter','integrated'),required=True)
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
        scope='scatter and valid local chain only; not distributed publication or timing')
    def save():
        (args.output/'probe.json').write_text(json.dumps(report,indent=2))
    save()
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec
    specs = (p('core',None,None),)*3+(p('core',None),)+(p('core',None,None),)*2
    def local(*xs):
        return tuple(x[None] for x in run_case(args.mode,*(x[0] for x in xs)))
    fn = jax.jit(jax.shard_map(local,mesh=mesh,in_specs=specs,
        out_specs=(p('core',None,None),)*2,check_vma=False))
    cases = [make_case(n) for n in (0,1,127,128,129)]
    if args.mode=='scatter':
        cases += [make_case(129,failure=f) for f in ('count_overflow','target_overflow')]
    for case in cases:
        inputs=case['inputs']
        row=dict(name=case['name'],exact=False,status='pending',
            input_sha256=hashlib.sha256(b''.join(x.tobytes() for x in inputs)).hexdigest(),
            expected_sha256=hashlib.sha256(case['expected'].tobytes()).hexdigest(),
            expected_error=case['expected_error'])
        report['cases'].append(row)
        save()
        xs=tuple(jax.device_put(np.broadcast_to(x,(8,)+x.shape).copy(),
            jax.sharding.NamedSharding(mesh,s)) for x,s in zip(inputs,specs))
        lowered=fn.lower(*xs)
        (args.output/f"{case['name']}.lowered.mlir").write_text(lowered.as_text())
        row['status']='compiling'
        save()
        exe=lowered.compile()
        (args.output/f"{case['name']}.hlo.txt").write_text(exe.as_text())
        actual,errors=map(np.asarray,jax.block_until_ready(exe(*xs)))
        mismatches=[int(np.count_nonzero(x!=case['expected'])) for x in actual]
        invalid=errors[:,0,0].tolist()
        row.update(status='executed',mismatches=mismatches,invalid=invalid,
            output_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in actual],
            exact=mismatches==[0]*8 and invalid==[case['expected_error']]*8)
        save()
    report['exact']=all(row['exact'] for row in report['cases'])
    save()
    raise SystemExit(0 if report['exact'] else 1)


if __name__=='__main__':
    main()
