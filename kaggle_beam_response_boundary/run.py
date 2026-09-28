"""Private physical response-boundary launcher; no JAX in the parent."""
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT_SHA = '1c1e9d82948ee7e2f01b119a9e57517a38076c2c'
CHECKOUT = Path('/tmp/TPUBeamSearch-response-boundary')
OUTPUT = Path('/kaggle/working/beam_response_boundary')


def main():
    subprocess.run((sys.executable, '-m', 'pip', 'install', '--quiet',
                    'jax[tpu]==0.10.2', 'jaxlib==0.10.2', 'libtpu==0.0.42.1'), check=True)
    subprocess.run(('git', 'clone', 'https://github.com/TryDotAtwo/TPUBeamSearch.git',
                    str(CHECKOUT)), check=True)
    subprocess.run(('git', 'checkout', '--detach', COMMIT_SHA), cwd=CHECKOUT, check=True)
    actual = subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=CHECKOUT,
                                     text=True).strip()
    dirty = subprocess.check_output(('git', 'status', '--porcelain'), cwd=CHECKOUT,
                                    text=True).strip()
    if actual != COMMIT_SHA or dirty:
        raise RuntimeError('pinned source checkout is not clean')
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64='False', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
               PYTHONPATH=os.pathsep.join((str(CHECKOUT), str(CHECKOUT / 'src'))))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    report = dict(source_sha=COMMIT_SHA, returncode=None)
    manifest = OUTPUT / 'process.json'
    manifest.write_text(json.dumps(report, indent=2))
    with (OUTPUT / 'process.log').open('w') as log:
        child = subprocess.run((sys.executable, '-m',
                                'benchmarks.beam_response_boundary_probe',
                                '--output', str(OUTPUT / 'probe')),
                               cwd=CHECKOUT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=False)
    report['returncode'] = child.returncode
    manifest.write_text(json.dumps(report, indent=2))
    if child.returncode:
        raise RuntimeError(f'boundary probe returned {child.returncode}; see process.log')


if __name__ == '__main__':
    main()
