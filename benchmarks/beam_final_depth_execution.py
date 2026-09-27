"""Physical eight-TPU one-depth final selection/materialization gate.

The fixture routes every selected parent request to a different TPU. No
timing or full beam throughput is measured. Output is checkpointed around
lowering, compilation and execution so native failures preserve evidence.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from tpu_beam_search.beam_final_depth import make_final_depth_call
from tpu_beam_search.beam_final_materialization_round import FinalMaterializationState


WORLD=8
STATE_LEN=150
STORAGE_WIDTH=160
KERNEL_WIDTH=256


def make_inputs():
    a=np.zeros((WORLD,1,8,128),np.uint32)
    b=np.zeros_like(a)
    a[:,:,6,:]=np.iinfo(np.uint32).max
    b[:,:,6,:]=np.iinfo(np.uint32).max
    controls=np.zeros((WORLD,1,8,128),np.uint32)
    threshold=np.full((WORLD,1),5,np.uint32)
    beam=np.zeros((WORLD,2,128),np.uint32)
    prior=np.zeros((WORLD,1,128),np.uint32)
    parents=np.zeros((WORLD,128,KERNEL_WIDTH),np.uint8)
    generators=np.broadcast_to(np.arange(KERNEL_WIDTH,dtype=np.int32),
                               (WORLD,1,KERNEL_WIDTH)).copy()
    for rank in range(WORLD):
        a[rank,0,6,0]=5
        a[rank,0,0,0]=rank+1
        a[rank,0,7,0]=((rank+1)%WORLD)<<16
        controls[rank,0,0,0]=1
        beam[rank,0,0]=WORLD
        parents[rank,0,:STATE_LEN]=rank+1
    state=FinalMaterializationState(
        np.zeros((WORLD,128,STORAGE_WIDTH),np.uint8),
        np.zeros((WORLD,1,5,128),np.uint32),
        np.zeros((WORLD,1,128),np.uint32),
        np.zeros((WORLD,2,128),np.uint32),
        np.zeros((WORLD,1,128),np.uint32),
        np.zeros((WORLD,2,128),np.uint32),
        np.zeros((WORLD,1,128),np.uint32))
    return a,b,controls,threshold,beam,prior,parents,generators,state


def expected():
    frontier=np.zeros((WORLD,128,STORAGE_WIDTH),np.uint8)
    history=np.zeros((WORLD,1,5,128),np.uint32)
    for rank in range(WORLD):
        source=(rank+1)%WORLD
        frontier[rank,0,:STATE_LEN]=source+1
        history[rank,0,:,0]=[0,0,source<<16,0,1]
    return frontier,history


def make_sharded_call(mesh):
    part=jax.sharding.PartitionSpec('core')
    local_state=FinalMaterializationState(*(part for _ in range(7)))
    depth=make_final_depth_call(mesh,state_len=STATE_LEN,move_count=1)

    def local(a,b,controls,threshold,beam,prior,parents,generators,state):
        single=FinalMaterializationState(*(x[0] for x in state))
        result,targets,keep,counts,re,he=depth(
            a[0],b[0],controls[0],threshold[0],beam[0],prior[0],
            parents[0],generators[0],single)
        wrapped=FinalMaterializationState(*(x[None] for x in result))
        return wrapped,targets[None],keep[None],counts[None],re[None],he[None]

    return jax.jit(jax.shard_map(local,mesh=mesh,
        in_specs=(part,part,part,part,part,part,part,part,local_state),
        out_specs=(local_state,part,part,part,part,part),check_vma=False))


def _sha(*arrays):
    digest=hashlib.sha256()
    for value in arrays:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    devices=jax.devices()
    if len(devices)!=WORLD or len({device.id for device in devices})!=WORLD or any(
            device.platform!='tpu' for device in devices):
        raise RuntimeError('requires eight distinct physical TPU devices')
    report={
        'status':'starting',
        'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runtime':{name:importlib.metadata.version(name)
                   for name in ('jax','jaxlib','libtpu')},
        'devices':[{'id':device.id,'kind':device.device_kind} for device in devices],
        'all_exact':False,
    }
    path=output/'final_depth.json'
    def save():
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    mesh=jax.sharding.Mesh(np.asarray(devices),('core',))
    part=jax.sharding.PartitionSpec('core')
    sharding=jax.sharding.NamedSharding(mesh,part)
    values=make_inputs()
    report['input_sha256']=_sha(*(values[:8]),*(tuple(values[8])))
    save()
    inputs=jax.tree.map(lambda x:jax.device_put(x,sharding),values)
    call=make_sharded_call(mesh)
    report['status']='lowering'
    save()
    lowered=call.lower(*inputs)
    (output/'lowered.mlir').write_text(lowered.as_text(),encoding='utf-8')
    report['status']='compiling'
    save()
    executable=lowered.compile()
    (output/'compiled.hlo.txt').write_text(executable.as_text(),encoding='utf-8')
    report['status']='executing'
    save()
    state,targets,keep,counts,response_error,history_error=(
        jax.block_until_ready(executable(*inputs)))
    frontier=np.asarray(state.frontier)
    history=np.asarray(state.history)
    want_frontier,want_history=expected()
    controls=(np.asarray(state.error),np.asarray(targets),
              np.asarray(response_error),np.asarray(history_error))
    keep_host=np.asarray(keep)
    counts_host=np.asarray(counts)
    controls_ok=(all(value.dtype==np.uint32 for value in
                     (*controls,keep_host,counts_host)) and
                 controls[1].shape==(WORLD,WORLD) and
                 np.array_equal(keep_host[:,0,0],np.full(WORLD,WORLD,np.uint32)) and
                 not keep_host[:,1,0].any() and
                 not counts_host[:,0,0].any() and
                 np.array_equal(counts_host[:,1,0],
                                np.full(WORLD,WORLD,np.uint32)))
    per_device=[bool(np.array_equal(frontier[i],want_frontier[i]) and
                     np.array_equal(history[i],want_history[i]) and
                     int(controls[1][i,0])==1 and
                     all(not np.asarray(x[i]).any() for x in
                         (controls[0],controls[2],controls[3])))
                for i in range(WORLD)]
    report.update(status='complete',exact_by_device=per_device,
        shape_ok=frontier.shape==want_frontier.shape and history.shape==want_history.shape,
        dtype_ok=frontier.dtype==np.uint8 and history.dtype==np.uint32,
        controls_ok=bool(controls_ok),
        frontier_sha256=_sha(frontier),history_sha256=_sha(history),
        expected_frontier_sha256=_sha(want_frontier),
        expected_history_sha256=_sha(want_history),
        keep=[int(x) for x in keep_host[:,0,0]],
        phase_counts=counts_host[:,:,0].astype(int).tolist())
    report['all_exact']=bool(all(per_device) and report['shape_ok'] and
                             report['controls_ok'] and
                             report['dtype_ok'] and
                             report['frontier_sha256']==report['expected_frontier_sha256'] and
                             report['history_sha256']==report['expected_history_sha256'])
    save()
    if not report['all_exact']:
        raise RuntimeError('eight-device final depth mismatch; see final_depth.json')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
