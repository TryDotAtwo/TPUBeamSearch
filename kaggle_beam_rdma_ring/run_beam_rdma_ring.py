"""Bootstrap immutable public source for the private eight-TPU RDMA probe."""
import os
from pathlib import Path
import subprocess
import sys
import json
import traceback

COMMIT_SHA = '1760770fe1898dd225227b5706fb42b73369623d'
CHECKOUT = Path('/tmp/TPUBeamSearch-rdma-ring')
OUTPUT = Path('/kaggle/working/beam_rdma_ring')


def main():
    subprocess.run((sys.executable, '-m', 'pip', 'install', '--quiet',
                    'jax[tpu]==0.10.2', 'jaxlib==0.10.2', 'libtpu==0.0.42.1'), check=True)
    subprocess.run(('git', 'clone', 'https://github.com/TryDotAtwo/TPUBeamSearch.git',
                    str(CHECKOUT)), check=True)
    subprocess.run(('git', 'checkout', '--detach', COMMIT_SHA), cwd=CHECKOUT, check=True)
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64='False', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
               PYTHONPATH=os.pathsep.join((str(CHECKOUT), str(CHECKOUT / 'src'))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = (('integrated_stream3_exchange', ('--mode', 'integrated')),)
    report = {'scope': 'compiled Stream3 split-to-wire-to-RDMA gate', 'source_sha': COMMIT_SHA,
              'cases': []}
    for name, extra in cases:
        destination = OUTPUT / name
        destination.mkdir(parents=True, exist_ok=True)
        command = (sys.executable, '-m', 'benchmarks.beam_rdma_ring_probe',
                   '--output', str(destination), *extra)
        row = {'name': name, 'status': 'started'}
        report['cases'].append(row)
        (OUTPUT / 'rdma_epoch_bundle.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        try:
            with (destination / 'process.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(
                    command, cwd=CHECKOUT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, text=True, timeout=120)
            row.update(status='complete' if completed.returncode == 0 else 'error',
                       returncode=completed.returncode)
        except subprocess.TimeoutExpired:
            row.update(status='timeout', error=traceback.format_exc())
        except Exception:
            row.update(status='error', error=traceback.format_exc())
        (OUTPUT / 'rdma_epoch_bundle.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        print(name, row['status'], flush=True)
    if not all(row['status'] == 'complete' for row in report['cases']):
        raise RuntimeError('one or more RDMA epoch cases failed')


if __name__ == '__main__':
    main()
