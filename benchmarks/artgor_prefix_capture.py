"""Materialized BF16 boundary capture; attribution requires an output control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import benchmarks.artgor_prefix_gate as gate
from benchmarks.artgor_prefix_shape import chunked_host
from benchmarks.artgor_input_trace import save_mismatch_rows

from benchmarks.artgor_pallas_same_suffix import (
    reference_embedding, reference_input_dense, reference_input_ln,
)


def captured_prefix(states, weights, architecture, *, include_invstd=False):
    """Return Dense, broadcast mean and original LN output in one executable.

    Extra outputs may change compiler fusion. Callers MUST compare slot 2
    with the untouched reference before attributing slots 0/1 to that reference.
    Keeping all slots BF16 avoids a widening trace eliding materialization.
    """
    embedded = reference_embedding(states, weights, architecture)
    dense = reference_input_dense(embedded, weights)
    mean = jnp.broadcast_to(jnp.mean(dense, axis=-1, keepdims=True), dense.shape)
    output = reference_input_ln(dense, weights, architecture)
    if include_invstd:
        variance = jnp.mean(jnp.square(dense-mean),axis=-1,keepdims=True)
        invstd = jax.lax.rsqrt(variance+architecture.LAYER_NORM_EPSILON)
        return jnp.stack((dense,mean,output,jnp.broadcast_to(invstd,dense.shape)),axis=1)
    return jnp.stack((dense, mean, output), axis=1)


def pallas_capture(states, weights, architecture, *, include_invstd=False):
    cfg = gate.config()
    embedded = gate.pallas_exact_embedding(states, weights, architecture, config=cfg)
    raw = gate.pallas_layernorm_dense(
        embedded, weights.input.dense.weight, weights.input.dense.bias,
        bm=cfg.input_bm, bk=128, bn=cfg.input_bn,
        dense_rounding='late', output_dtype=jnp.float32)
    mean = gate.mean_buffer(raw, pallas=True, order='lanes_tree', bm=cfg.input_bm)
    output = gate.external_mean_ln(raw, mean, weights.input.normalization.scale,
                                  weights.input.normalization.bias,
                                  epsilon=architecture.LAYER_NORM_EPSILON, bm=cfg.input_bm)
    dense=raw.astype(jnp.bfloat16)
    if include_invstd:
        from benchmarks.artgor_invstd_capture import invstd_buffer
        invstd=invstd_buffer(dense,mean,epsilon=architecture.LAYER_NORM_EPSILON)
        return jnp.stack((dense,mean,output,invstd),axis=1)
    return jnp.stack((dense, mean, output), axis=1)


def run(dataset, output, *, include_invstd=False):
    output.mkdir(parents=True, exist_ok=True)
    path = output/'artgor_prefix_capture.json'
    report = dict(status='running', schema_version=1, comparisons={},
                  scope='captured_input_prefix_only_no_speed_claim',include_invstd=include_invstd)
    gate.checkpoint(path, report)
    try:
        devices = jax.devices()
        if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
            raise RuntimeError('requires exactly eight TPU devices')
        sys.path.insert(0, str(dataset))
        from jax_model import load_params_from_pt
        cp = dataset/'q555_2k_BEST.pt'
        params = load_params_from_pt(cp)
        architecture = gate.Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=150)
        weights = gate.layernorm_stream1_weights_from_artgor_params(params, architecture)
        packed = gate.prepare_pallas_exact_weights(weights, architecture)
        mesh = Mesh(np.asarray(devices), ('core',))
        spec = P('core', None)
        sharding = NamedSharding(mesh, spec)
        wd, pd = gate._replicate(weights, mesh), gate._replicate(packed, mesh)
        puzzle_path = dataset/'puzzle_info.json'
        puzzle = gate.load_puzzle(puzzle_path, state_len=150, move_count=30)
        host = gate._make_states(puzzle, 'legal', 42, 8*16384)
        previous = json.loads((Path(__file__).resolve().parents[1]/
            'test_results/artgor_prefix_gate_v1/artgor_prefix_gate/artgor_prefix_gate.json').read_text())
        previous_case = next(c for c in previous['cases'] if c['batch_per_device']==16384
                             and c['kind']=='legal' and c['seed']==42 and c['order']=='lanes_tree')
        input_hash = gate._array_sha256(host)
        if input_hash != previous_case['input_sha256']:
            raise RuntimeError('input corpus hash changed')
        report['context'] = dict(source_commit=subprocess.check_output(
            ('git','rev-parse','HEAD'), text=True).strip(), runtime=gate.runtime_inventory(),
            input_sha256=input_hash, checkpoint_sha256=gate.sha256_file(cp),
            model_source_sha256=gate.sha256_file(dataset/'jax_model.py'),
            puzzle_sha256=gate.sha256_file(puzzle_path), local_batches=[16384,256],
            kind='legal', seed=42)

        def compare(label, left, right):
            result = gate.compare_prefix(left, right)
            report['comparisons'][label] = result
            save_mismatch_rows(output/f'{label}.npz', host,
                               left.astype(np.float32), right.astype(np.float32))
            gate.checkpoint(path, report)
            return result

        calls = {}
        for label, function, example, resident in (
            ('jax', lambda x,w: gate.reference_hidden_after_depth(x,w,architecture,0), weights,wd),
            ('pallas', lambda x,w: gate.pallas_prefix(x,w,architecture,order='lanes_tree'),packed,pd),
            ('jax_capture',lambda x,w: captured_prefix(x,w,architecture,include_invstd=include_invstd),weights,wd),
            ('pallas_capture',lambda x,w: pallas_capture(x,w,architecture,include_invstd=include_invstd),packed,pd),
        ):
            calls[label] = (gate._mapped(function,mesh=mesh,input_spec=spec,weights_example=example),resident)
        # The same external-mean remainder is used for every substitution.
        def remainder(values, w):
            return gate.external_mean_ln(values[:,0].astype(jnp.float32), values[:,1],
                w.input.normalization.scale,w.input.normalization.bias,
                epsilon=architecture.LAYER_NORM_EPSILON,bm=128)
        remainder_call = gate._mapped(remainder,mesh=mesh,input_spec=spec,weights_example=packed)
        report['attribution_controls'] = {}
        large_captures = {}
        for size in (16384,256):
            arrays = {}
            for label,(call,resident) in calls.items():
                def operation(chunk):
                    return jax.block_until_ready(call(jax.device_put(chunk,sharding),resident))
                arrays[label] = chunked_host(host,operation,chunk_rows=size)
                sample = host.reshape(8,16384,150)[:,:size].reshape(8*size,150)
                lowered = call.lower(jax.device_put(sample,sharding),resident)
                (output/f'{label}_{size}.compiled.txt').write_text(lowered.compile().as_text(),encoding='utf-8')
                (output/f'{label}_{size}.stablehlo.txt').write_text(str(lowered.compiler_ir(dialect='stablehlo')),encoding='utf-8')
            j, p = arrays['jax_capture'], arrays['pallas_capture']
            if size == 16384:
                large_captures = {'jax': j, 'pallas': p}
            else:
                for source, captured in (('jax',j),('pallas',p)):
                    names=('dense','mean','output','invstd') if include_invstd else ('dense','mean','output')
                    for slot, name in enumerate(names):
                        compare(f'{source}_shape_{name}',large_captures[source][:,slot],captured[:,slot])
                large_captures.clear()
            control = compare(f'{size}_jax_capture_control',arrays['jax'],j[:,2])
            native = compare(f'{size}_pallas_capture_control',arrays['pallas'],p[:,2])
            compare(f'{size}_dense',j[:,0],p[:,0])
            compare(f'{size}_mean',j[:,1],p[:,1])
            compare(f'{size}_untouched',arrays['jax'],arrays['pallas'])
            if size == 16384:
                c = report['comparisons'][f'{size}_untouched']
                report['reproduces_previous_large_hashes'] = all(
                    c[key] == previous_case['comparison'][key]
                    for key in ('reference_sha256','candidate_sha256'))
                if not report['reproduces_previous_large_hashes']:
                    raise RuntimeError('untouched large outputs changed since shape diagnostic')
            # Run even if capture failed, but explicitly disallow causal attribution.
            zero_exact = False
            for dense_label,dense in (('pallas',p[:,0]),('jax',j[:,0])):
                for mean_label,mean in (('pallas',p[:,1]),('jax',j[:,1])):
                    inputs = np.stack((dense,mean),axis=1)
                    result = chunked_host(inputs,lambda chunk: jax.block_until_ready(
                        remainder_call(jax.device_put(chunk,sharding),pd)),chunk_rows=size)
                    label = f'{size}_dense_{dense_label}_mean_{mean_label}'
                    compare(label,arrays['jax'],result)
                    if dense_label == mean_label == 'pallas':
                        zero_exact = compare(f'{size}_remainder_zero_control',arrays['pallas'],result)['exact']
                        sample = inputs.reshape(8,16384,*inputs.shape[1:])[:,:size].reshape(8*size,*inputs.shape[1:])
                        lowered = remainder_call.lower(jax.device_put(sample,sharding),pd)
                        (output/f'remainder_{size}.compiled.txt').write_text(lowered.compile().as_text(),encoding='utf-8')
                        (output/f'remainder_{size}.stablehlo.txt').write_text(str(lowered.compiler_ir(dialect='stablehlo')),encoding='utf-8')
                    del inputs, result
            report['attribution_controls'][str(size)] = dict(
                valid=control['exact'] and native['exact'] and zero_exact,
                capture_exact=control['exact'],pallas_capture_exact=native['exact'],
                remainder_zero_exact=zero_exact)
            if include_invstd:
                from benchmarks.artgor_invstd_capture import invstd_buffer, external_invstd_affine
                def affine(values,w):
                    return external_invstd_affine(values[:,0],values[:,1],values[:,2],
                        w.input.normalization.scale,w.input.normalization.bias)
                affine_call=gate._mapped(affine,mesh=mesh,input_spec=spec,weights_example=packed)
                inv_call=gate._mapped(lambda values,w:invstd_buffer(values[:,0],values[:,1],
                    epsilon=architecture.LAYER_NORM_EPSILON),mesh=mesh,input_spec=spec,weights_example=packed)
                def evaluate_aux(label,call,values):
                    result=chunked_host(values,lambda chunk:jax.block_until_ready(
                        call(jax.device_put(chunk,sharding),pd)),chunk_rows=size)
                    sample=values.reshape(8,16384,*values.shape[1:])[:,:size].reshape(8*size,*values.shape[1:])
                    lowered=call.lower(jax.device_put(sample,sharding),pd)
                    (output/f'{label}_{size}.compiled.txt').write_text(lowered.compile().as_text(),encoding='utf-8')
                    (output/f'{label}_{size}.stablehlo.txt').write_text(str(lowered.compiler_ir(dialect='stablehlo')),encoding='utf-8')
                    return result
                compare(f'{size}_invstd_capture',j[:,3],p[:,3])
                native_out=evaluate_aux('affine',affine_call,np.stack((p[:,0],p[:,1],p[:,3]),axis=1))
                native_affine=compare(f'{size}_affine_native_zero',arrays['pallas'],native_out)['exact']
                del native_out
                # Recompute Pallas invstd with JAX's actual Dense/mean. Otherwise
                # changing mean AND invstd would confound the remaining two rows.
                fixed_inputs=np.stack((j[:,0],j[:,1]),axis=1)
                fixed_inv=evaluate_aux('fixed_mean_invstd',inv_call,fixed_inputs)
                fixed_reference=evaluate_aux('fixed_mean_remainder',remainder_call,fixed_inputs)
                compare(f'{size}_fixed_mean_invstd',j[:,3],fixed_inv)
                fixed_zero=False
                for label,inv in (('pallas',fixed_inv),('jax',j[:,3])):
                    result=evaluate_aux(f'fixed_mean_affine_{label}',affine_call,
                        np.stack((j[:,0],j[:,1],inv),axis=1))
                    compare(f'{size}_fixed_mean_affine_{label}',arrays['jax'],result)
                    if label=='pallas':
                        fixed_zero=compare(f'{size}_fixed_mean_split_zero',fixed_reference,result)['exact']
                    del result
                controls=report['attribution_controls'][str(size)]
                controls.update(invstd_native_zero_exact=native_affine,fixed_mean_split_zero_exact=fixed_zero)
                controls['valid']=controls['valid'] and native_affine and fixed_zero
                del fixed_inputs,fixed_inv,fixed_reference
            del arrays,j,p
            gate.checkpoint(path,report)
        report['status']='complete'
        gate.checkpoint(path,report)
    except Exception as error:
        report.update(status='error',error=str(error),traceback=traceback.format_exc())
        gate.checkpoint(path,report)
        raise
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-invstd',action='store_true')
    args=parser.parse_args()
    print(json.dumps({'status':run(gate._dataset_path(args.dataset),args.output,include_invstd=args.include_invstd)['status']}))
