"""Bootstrap immutable source for the private eight-TPU external-sort gate."""
import os
from pathlib import Path
import subprocess
import sys


COMMIT_SHA = '42037f44ea24e2303ae60464d00cfba33e30fd28'
CHECKOUT = Path('/tmp/TPUBeamSearch-external-sort')
OUTPUT = Path('/kaggle/working/beam_external_sort')


def main():
    subprocess.run((sys.executable, '-m', 'pip', 'install', '--quiet',
                    'jax[tpu]==0.10.2', 'jaxlib==0.10.2',
                    'libtpu==0.0.42.1'), check=True)
    subprocess.run(('git', 'clone',
                    'https://github.com/TryDotAtwo/TPUBeamSearch.git',
                    str(CHECKOUT)), check=True)
    subprocess.run(('git', 'checkout', '--detach', COMMIT_SHA),
                   cwd=CHECKOUT, check=True)
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64='False', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
               PYTHONPATH=os.pathsep.join((str(CHECKOUT),
                                           str(CHECKOUT / 'src'))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subprocess.run((sys.executable, '-m',
                    'benchmarks.beam_external_sort_probe',
                    '--output', str(OUTPUT)), cwd=CHECKOUT, env=env,
                   check=True)


if __name__ == '__main__':
    main()
