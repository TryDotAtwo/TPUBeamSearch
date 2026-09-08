"""Private routed response gate; parent deliberately imports no JAX."""
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT_SHA='8dc27ca763764c8485988c06210fddfc7c0f5854'
CHECKOUT=Path('/tmp/TPUBeamSearch-response-gate')
OUTPUT=Path('/kaggle/working/beam_response_epoch')


def main():
    subprocess.run((sys.executable,'-m','pip','install','--quiet',
        'jax[tpu]==0.10.2','jaxlib==0.10.2','libtpu==0.0.42.1'),check=True)
    subprocess.run(('git','clone','https://github.com/TryDotAtwo/TPUBeamSearch.git',str(CHECKOUT)),check=True)
    subprocess.run(('git','checkout','--detach',COMMIT_SHA),cwd=CHECKOUT,check=True)
    env=os.environ.copy()
    env.update(JAX_ENABLE_X64='False',PYTHONUNBUFFERED='1',
        XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        PYTHONPATH=os.pathsep.join((str(CHECKOUT),str(CHECKOUT/'src'))))
    OUTPUT.mkdir(parents=True,exist_ok=False)
    report=dict(source_sha=COMMIT_SHA,returncode=None)
    manifest=OUTPUT/'process.json'
    manifest.write_text(json.dumps(report,indent=2))
    with (OUTPUT/'process.log').open('w') as log:
        child=subprocess.run((sys.executable,'-m','benchmarks.beam_response_epoch_probe',
            '--output',str(OUTPUT/'probe')),cwd=CHECKOUT,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=False)
    report['returncode']=child.returncode
    manifest.write_text(json.dumps(report,indent=2))
    if child.returncode:
        raise RuntimeError(f'response gate child returned {child.returncode}; see process.log')


if __name__=='__main__':
    main()
