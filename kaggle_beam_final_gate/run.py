"""Prepared private final gate; submit only after the S5 TPU session terminates."""
import os
from pathlib import Path
import subprocess
import sys

COMMIT_SHA = 'be939296d25133e20ae921ddb4547aaa4a8c457e'
CHECKOUT = Path('/tmp/TPUBeamSearch-final-gate')
OUTPUT = Path('/kaggle/working/beam_final_isolation')


def main():
    subprocess.run((sys.executable,'-m','pip','install','--quiet',
        'jax[tpu]==0.10.2','jaxlib==0.10.2','libtpu==0.0.42.1'),check=True)
    subprocess.run(('git','clone','https://github.com/TryDotAtwo/TPUBeamSearch.git',str(CHECKOUT)),check=True)
    subprocess.run(('git','checkout','--detach',COMMIT_SHA),cwd=CHECKOUT,check=True)
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64='False',PYTHONUNBUFFERED='1',
        XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
        PYTHONPATH=os.pathsep.join((str(CHECKOUT),str(CHECKOUT/'src'))))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    subprocess.run((sys.executable,'-m','benchmarks.beam_final_isolation_probe',
        '--mode','select_reduce','--output',str(OUTPUT/'select_reduce')),cwd=CHECKOUT,env=env,check=True)


if __name__ == '__main__':
    main()
