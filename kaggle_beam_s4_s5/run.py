"""Pinned private S4/S5 isolated physical gates, V1."""
import os
from pathlib import Path
import subprocess
import sys

COMMIT_SHA = 'c35734079bb117322fbaaee72fd1d4e20191eab8'
CHECKOUT = Path('/tmp/TPUBeamSearch-s4-s5')
OUTPUT = Path('/kaggle/working/beam_s4_s5')


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
    subprocess.run((sys.executable,'-m','benchmarks.beam_s4_s5_bundle',
        '--output',str(OUTPUT)),cwd=CHECKOUT,env=env,check=True)


if __name__ == '__main__':
    main()
