"""Same states, two compilation shapes; inference-only numerical diagnosis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import benchmarks.artgor_prefix_gate as gate
from benchmarks.artgor_input_trace import save_mismatch_rows


def chunked_host(states, operation, *, devices=8, chunk_rows=256):
    if devices <= 0 or chunk_rows <= 0 or len(states) % devices or not len(states):
        raise ValueError('invalid device partition')
    local = len(states) // devices
    if local % chunk_rows:
        raise ValueError('invalid chunk partition')
    partition = np.asarray(states).reshape(devices, local, *states.shape[1:])
    output = None
    for start in range(0, local, chunk_rows):
        chunk = partition[:,start:start+chunk_rows].reshape(devices*chunk_rows,*states.shape[1:])
        result = np.asarray(operation(chunk))
        if len(result) != devices*chunk_rows:
            raise ValueError('operation changed row count')
        if output is None:
            output = np.empty((devices,local,*result.shape[1:]),dtype=result.dtype)
        output[:,start:start+chunk_rows] = result.reshape(devices,chunk_rows,*result.shape[1:])
    return output.reshape(len(states),*output.shape[2:])


def run(dataset, output):
    output.mkdir(parents=True,exist_ok=True)
    path = output/'artgor_prefix_shape.json'
    report = dict(status='running',schema_version=1,comparisons={},scope='same_state_prefix_only')
    gate.checkpoint(path,report)
    try:
        devices = jax.devices()
        if len(devices)!=8 or any(d.platform!='tpu' for d in devices):
            raise RuntimeError('requires eight TPU devices')
        sys.path.insert(0,str(dataset))
        from jax_model import load_params_from_pt
        cp = dataset/'q555_2k_BEST.pt'
        params = load_params_from_pt(cp)
        architecture = gate.Stream1Architecture.from_artgor_params(params,STATE_STORAGE_LEN=150)
        weights = gate.layernorm_stream1_weights_from_artgor_params(params,architecture)
        packed = gate.prepare_pallas_exact_weights(weights,architecture)
        mesh = Mesh(np.asarray(devices),('core',))
        spec = P('core',None)
        sharding = NamedSharding(mesh,spec)
        wd,pd = gate._replicate(weights,mesh),gate._replicate(packed,mesh)
        reference = gate._mapped(lambda x,w:gate.reference_hidden_after_depth(x,w,architecture,0),
                                 mesh=mesh,input_spec=spec,weights_example=weights)
        candidate = gate._mapped(lambda x,w:gate.pallas_prefix(x,w,architecture,order='lanes_tree'),
                                 mesh=mesh,input_spec=spec,weights_example=packed)
        puzzle = gate.load_puzzle(dataset/'puzzle_info.json',state_len=150,move_count=30)
        host = gate._make_states(puzzle,'legal',42,8*16384)
        previous = json.loads((Path(__file__).resolve().parents[1]/'test_results/artgor_prefix_gate_v1/artgor_prefix_gate/artgor_prefix_gate.json').read_text())
        previous_case = next(c for c in previous['cases'] if c['batch_per_device']==16384 and c['kind']=='legal' and c['seed']==42 and c['order']=='lanes_tree')
        input_hash = gate._array_sha256(host)
        if input_hash != previous_case['input_sha256']:
            raise RuntimeError('input hash differs from the first failing gate corpus')
        report['context'] = dict(source_commit=subprocess.check_output(('git','rev-parse','HEAD'),text=True).strip(),
                                input_sha256=input_hash,runtime=gate.runtime_inventory(),
                                checkpoint_sha256=gate.sha256_file(cp),model_source_sha256=gate.sha256_file(dataset/'jax_model.py'),
                                full_local_batch=16384,chunk_local_batch=256,kind='legal',seed=42)
        gate.checkpoint(path,report)
        states = jax.device_put(host,sharding)
        arrays = {}
        for label,call,w in (('jax',reference,wd),('pallas',candidate,pd)):
            arrays[label+'_large'] = np.asarray(jax.block_until_ready(call(states,w)))
            def operation(chunk):
                return jax.block_until_ready(call(jax.device_put(chunk,sharding),w))
            arrays[label+'_chunked'] = chunked_host(host,operation)
            for size in (16384,256):
                shaped = host.reshape(8,16384,150)[:,:size].reshape(8*size,150)
                lowered = call.lower(jax.device_put(shaped,sharding),w)
                (output/f'{label}_{size}.compiled.txt').write_text(lowered.compile().as_text(),encoding='utf-8')
                (output/f'{label}_{size}.stablehlo.txt').write_text(str(lowered.compiler_ir(dialect='stablehlo')),encoding='utf-8')
        for label,left,right in (
            ('jax_shape','jax_large','jax_chunked'),
            ('pallas_shape','pallas_large','pallas_chunked'),
            ('large_matched','jax_large','pallas_large'),
            ('small_matched','jax_chunked','pallas_chunked'),
        ):
            report['comparisons'][label] = gate.compare_prefix(arrays[left],arrays[right])
            save_mismatch_rows(output/f'{label}_mismatches.npz',host,
                               arrays[left].astype(np.float32),arrays[right].astype(np.float32))
            gate.checkpoint(path,report)
        report['reproduces_previous_large_hashes'] = (
            report['comparisons']['large_matched']['reference_sha256']==previous_case['comparison']['reference_sha256']
            and report['comparisons']['large_matched']['candidate_sha256']==previous_case['comparison']['candidate_sha256'])
        report['status']='complete'
        gate.checkpoint(path,report)
    except Exception as error:
        report.update(status='error',error=str(error),traceback=traceback.format_exc())
        gate.checkpoint(path,report)
        raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({'status':run(gate._dataset_path(args.dataset),args.output)['status']}))
