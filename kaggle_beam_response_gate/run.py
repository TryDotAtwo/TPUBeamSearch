"""Private routed response gate; parent deliberately imports no JAX."""
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT_SHA='d5c8a45f02a9a71e05da2f5584c887fbf968bda3'
CHECKOUT=Path('/tmp/TPUBeamSearch-response-gate')
OUTPUT=Path('/kaggle/working/beam_response_epoch')


def main():
    subprocess.run((sys.executable,'-m','pip','install','--quiet',
        'jax[tpu]==0.10.2','jaxlib==0.10.2','libtpu==0.0.42.1'),check=True)
    subprocess.run(('git','clone','https://github.com/TryDotAtwo/TPUBeamSearch.git',str(CHECKOUT)),check=True)
    subprocess.run(('git','checkout','--detach',COMMIT_SHA),cwd=CHECKOUT,check=True)
    actual=subprocess.check_output(('git','rev-parse','HEAD'),cwd=CHECKOUT,text=True).strip()
    dirty=subprocess.check_output(('git','status','--porcelain'),cwd=CHECKOUT,text=True).strip()
    if actual!=COMMIT_SHA or dirty:
        raise RuntimeError('pinned source checkout is not clean')
    env=os.environ.copy()
    env.update(JAX_ENABLE_X64='False',PYTHONUNBUFFERED='1',
        XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        PYTHONPATH=os.pathsep.join((str(CHECKOUT),str(CHECKOUT/'src'))))
    OUTPUT.mkdir(parents=True,exist_ok=False)
    report=dict(source_sha=COMMIT_SHA,returncode=None)
    manifest=OUTPUT/'process.json'
    manifest.write_text(json.dumps(report,indent=2))
    with (OUTPUT/'process.log').open('w') as log:
        child=subprocess.run((sys.executable,'-m','benchmarks.beam_response_followup_bundle',
            '--output',str(OUTPUT/'bundle')),cwd=CHECKOUT,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=False)
    report['returncode']=child.returncode
    manifest.write_text(json.dumps(report,indent=2))
    if child.returncode:
        raise RuntimeError(f'response follow-up returned {child.returncode}; see process.log')


if __name__=='__main__':
    main()
