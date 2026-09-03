"""Six-corpus large-batch correctness gate for the composed Pallas input prefix."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from benchmarks.artgor_input_trace import mean_buffer, external_mean_ln
from benchmarks.artgor_pallas_same_suffix import config, reference_hidden_after_depth, _mapped
from benchmarks.artgor_pallas_exact_diagnostic import _make_states, audit_all_pallas_hlo
from benchmarks.artgor_exact_notebook_validation import _dataset_path, _replicate, checkpoint, _array_sha256
from benchmarks.layernorm_quality import load_puzzle
from benchmarks.stream1_layernorm_arithmetic import runtime_inventory, sha256_file
from tpu_beam_search.stream1_architecture import Stream1Architecture
from tpu_beam_search.stream1_layernorm_reference import layernorm_stream1_weights_from_artgor_params
from tpu_beam_search.stream1_layernorm_pallas_exact import prepare_pallas_exact_weights, pallas_exact_embedding
from tpu_beam_search.stream1_layernorm_pallas import pallas_layernorm_dense

ORDERS = ('lanes_tree', 'tiles_serial', 'tiles_tree')
CORPORA = tuple(('legal', s) for s in (42,142,242)) + tuple(('stress', s) for s in (43,143,243))
BATCHES = (16384, 32768)


def compare_prefix(reference, candidate, *, chunk_rows=4096):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError('prefix shape/dtype mismatch')
    hashes = [hashlib.sha256(), hashlib.sha256()]
    count, first, finite = 0, None, True
    for start in range(0, reference.shape[0], chunk_rows):
        a, b = [np.asarray(x[start:start+chunk_rows]) for x in (reference, candidate)]
        av, bv = [np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, x.dtype.itemsize) for x in (a,b)]
        hashes[0].update(av.tobytes())
        hashes[1].update(bv.tobytes())
        unequal = np.any(av != bv, axis=-1)
        count += int(np.count_nonzero(unequal))
        if first is None and np.any(unequal):
            local = np.unravel_index(np.argmax(unequal), unequal.shape)
            first = [start + int(local[0]), int(local[1])]
        finite = finite and bool(np.isfinite(a.astype(np.float32)).all() and np.isfinite(b.astype(np.float32)).all())
    return dict(exact=finite and count == 0, finite=finite, mismatch_count=count,
                first_mismatch=first, shape=list(reference.shape), dtype=str(reference.dtype),
                reference_sha256=hashes[0].hexdigest(), candidate_sha256=hashes[1].hexdigest())


def pallas_prefix(states, weights, architecture, *, order, interpret=False):
    cfg = config()
    embedded = pallas_exact_embedding(states, weights, architecture, config=cfg, interpret=interpret)
    raw = pallas_layernorm_dense(embedded, weights.input.dense.weight, weights.input.dense.bias,
                                bm=cfg.input_bm, bk=128, bn=cfg.input_bn,
                                dense_rounding='late', output_dtype=jnp.float32, interpret=interpret)
    mean = mean_buffer(raw, pallas=True, order=order, bm=cfg.input_bm, interpret=interpret)
    return external_mean_ln(raw, mean, weights.input.normalization.scale, weights.input.normalization.bias,
                            epsilon=architecture.LAYER_NORM_EPSILON, bm=cfg.input_bm, interpret=interpret)


def run(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'artgor_prefix_gate.json'
    report = dict(status='running', schema_version=1, cases=[], hlo={}, scope='input_prefix_only_no_speed_claim')
    try:
        devices = jax.devices()
        if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
            raise RuntimeError('requires exactly eight TPU devices')
        sys.path.insert(0, str(dataset))
        from jax_model import load_params_from_pt
        cp = dataset / 'q555_2k_BEST.pt'
        params = load_params_from_pt(cp)
        architecture = Stream1Architecture.from_artgor_params(params, STATE_STORAGE_LEN=150)
        weights = layernorm_stream1_weights_from_artgor_params(params, architecture)
        packed = prepare_pallas_exact_weights(weights, architecture)
        mesh = Mesh(np.asarray(devices), ('core',))
        spec = P('core', None)
        wd, pd = _replicate(weights, mesh), _replicate(packed, mesh)
        reference = _mapped(lambda x,w: reference_hidden_after_depth(x,w,architecture,0),
                            mesh=mesh, input_spec=spec, weights_example=weights)
        runners = {order: _mapped(lambda x,w,o=order: pallas_prefix(x,w,architecture,order=o),
                                  mesh=mesh, input_spec=spec, weights_example=packed) for order in ORDERS}
        puzzle = load_puzzle(dataset / 'puzzle_info.json', state_len=150, move_count=30)
        report['context'] = dict(source_commit=subprocess.check_output(('git','rev-parse','HEAD'),text=True).strip(),
                                 runtime=runtime_inventory(), checkpoint_sha256=sha256_file(cp),
                                 model_source_sha256=sha256_file(dataset / 'jax_model.py'),
                                 batches=list(BATCHES), corpora=list(CORPORA))
        checkpoint(path, report)
        for batch in BATCHES:
            for kind, seed in CORPORA:
                host = _make_states(puzzle, kind, seed, batch*8)
                states = jax.device_put(host, NamedSharding(mesh,spec))
                expected = jax.block_until_ready(reference(states,wd))
                input_hash = _array_sha256(host)
                for order, runner in runners.items():
                    row = dict(batch_per_device=batch,kind=kind,seed=seed,order=order,input_sha256=input_hash)
                    phase = 'compile'
                    try:
                        key = f'{batch}_{order}'
                        lowered = runner.lower(states,pd)
                        compiled = lowered.compile()
                        if key not in report['hlo']:
                            stable = str(lowered.compiler_ir(dialect='stablehlo'))
                            report['hlo'][key] = audit_all_pallas_hlo(stable, ('embedding','dense','mean','remainder'))
                            (output / f'{key}.stablehlo.txt').write_text(stable,encoding='utf-8')
                            (output / f'{key}.compiled.txt').write_text(compiled.as_text(),encoding='utf-8')
                            ref_lowered = reference.lower(states,wd)
                            (output / f'{batch}_reference.compiled.txt').write_text(ref_lowered.compile().as_text(),encoding='utf-8')
                        phase = 'execute'
                        actual = jax.block_until_ready(compiled(states,pd))
                        row.update(status='complete',comparison=compare_prefix(expected,actual))
                        row['passes'] = row['comparison']['exact'] and report['hlo'][key]['passes']
                        del actual
                    except Exception as error:
                        row.update(status='error',phase=phase,error_type=type(error).__name__,error=str(error),passes=False)
                    report['cases'].append(row)
                    checkpoint(path,report)
                del expected, states, host
                gc.collect()
        report['status'] = 'complete'
        report['eligible_orders'] = [o for o in ORDERS if all(c['passes'] for c in report['cases'] if c['order']==o)]
        checkpoint(path,report)
    except Exception as error:
        report.update(status='error',error=str(error),traceback=traceback.format_exc())
        checkpoint(path,report)
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps({'status':run(_dataset_path(args.dataset),args.output)['status']}))
