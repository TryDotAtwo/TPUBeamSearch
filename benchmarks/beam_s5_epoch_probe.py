"""Twenty state-carrying serialized epochs on eight physical TPU devices."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from benchmarks.beam_external_dedup_probe import digest
from tpu_beam_search.beam_s5_epoch import make_s5_epoch_call


def fixtures():
    hist = np.zeros((8,2,128),np.uint32)
    hist[5,0,5] = 9
    hist_b = np.zeros_like(hist)
    hist_active = np.zeros((8,1,128),np.uint32)
    a,b = np.zeros_like(hist),np.zeros_like(hist)
    active = np.zeros_like(hist_active)
    beam = np.zeros_like(hist)
    beam[:,0,0] = 3
    state = np.zeros((8,4,128),np.uint32)
    initial = (hist,hist_b,hist_active,a.copy(),b.copy(),active.copy(),beam,state.copy())
    steps = []
    for repeat in range(2):
        for phase in range(10):
            force = np.zeros((8,1),np.uint32)
            if phase == 9:
                force[:,0] = 1
            elif phase:
                force[phase-1,0] = 1
            if phase:
                inactive = b if active[0,0,0] == 0 else a
                inactive[:,0,0],inactive[:,1,0] = 5,1
                active[:,0,0] ^= 1
                state[:,1,0] += 1
            steps.append((force,tuple(x.copy() for x in (a,b,active,state))))
    return initial,steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True,exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        scope='serialized S5 state-carry epochs, frozen histograms; not concurrent writer/reader or full beam',
        exact=False,cases=[])
    def save():
        (output/'s5_epoch.json').write_text(json.dumps(report,indent=2))
    save()
    initial,steps = fixtures()
    mesh = jax.sharding.Mesh(np.asarray(devices),('core',))
    p = jax.sharding.PartitionSpec
    specs = tuple(p('core',*(None for _ in x.shape[1:])) for x in (*initial,steps[0][0]))
    call = make_s5_epoch_call(mesh,bins=128,period=3)
    def local(*args):
        return tuple(x[None] for x in call(*(x[0] for x in args)))
    fn = jax.jit(jax.shard_map(local,mesh=mesh,in_specs=specs,
        out_specs=(specs[3],specs[4],specs[5],specs[7]),check_vma=False))
    args = tuple(jax.device_put(x,jax.sharding.NamedSharding(mesh,s))
                 for x,s in zip((*initial,steps[0][0]),specs))
    start = time.perf_counter()
    exe = fn.lower(*args).compile()
    report['compile_seconds'] = time.perf_counter()-start
    (output/'s5_epoch.hlo.txt').write_text(exe.as_text())
    save()
    for index,(force,expected) in enumerate(steps):
        args = (*args[:8],jax.device_put(force,jax.sharding.NamedSharding(mesh,specs[8])))
        row = dict(index=index,input_sha256=digest(tuple(np.asarray(x) for x in args)),
                   expected_sha256=digest(expected),exact=False)
        report['cases'].append(row)
        save()
        print('EPOCH',index,flush=True)
        result = jax.block_until_ready(exe(*args))
        actual = tuple(np.asarray(x) for x in result)
        mismatch = [int(np.count_nonzero(x != y)) for x,y in zip(actual,expected,strict=True)]
        row.update(exact=not any(mismatch),mismatches=mismatch,output_sha256=digest(actual))
        save()
        if any(mismatch):
            raise RuntimeError(f'epoch{index} mismatch')
        # Feed actual device outputs forward, not the host expected state.
        args = (*args[:3],*result[:3],args[6],result[3],args[8])
    report['exact'] = True
    save()


if __name__ == '__main__':
    main()
