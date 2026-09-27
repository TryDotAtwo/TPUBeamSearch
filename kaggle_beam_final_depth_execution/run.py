"""Private pinned GitHub runner for the physical final-depth gate."""
import json
import os
from pathlib import Path
import subprocess
import sys


COMMIT_SHA='b4ade889bdebb9b6020468709d490698c3768058'
CHECKOUT=Path('/tmp/TPUBeamSearch-final-depth-gate')
OUTPUT=Path('/kaggle/working/final_depth_execution')


def main():
    subprocess.run((sys.executable,'-m','pip','install','--quiet',
        'jax[tpu]==0.10.2','jaxlib==0.10.2','libtpu==0.0.42.1'),check=True)
    subprocess.run(('git','clone','https://github.com/TryDotAtwo/TPUBeamSearch.git',
                    str(CHECKOUT)),check=True)
    subprocess.run(('git','checkout','--detach',COMMIT_SHA),cwd=CHECKOUT,check=True)
    actual=subprocess.check_output(('git','rev-parse','HEAD'),cwd=CHECKOUT,text=True).strip()
    dirty=subprocess.check_output(('git','status','--porcelain'),cwd=CHECKOUT,text=True).strip()
    if actual!=COMMIT_SHA or dirty:
        raise RuntimeError('pinned source checkout is not clean')
    env=os.environ.copy()
    env.update(JAX_ENABLE_X64='False',PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join((str(CHECKOUT),str(CHECKOUT/'src'))))
    OUTPUT.mkdir(parents=True,exist_ok=False)
    report={'source_sha':COMMIT_SHA,'returncode':None}
    manifest=OUTPUT/'process.json'
    manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUTPUT/'process.log').open('w',encoding='utf-8') as log:
        result=subprocess.run((sys.executable,'-m',
            'benchmarks.beam_final_depth_execution','--output',
            str(OUTPUT/'result')),cwd=CHECKOUT,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=False)
    report['returncode']=result.returncode
    manifest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'final depth gate returned {result.returncode}; see process.log')


if __name__=='__main__':
    main()
