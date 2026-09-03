"""Run immutable selector probes in isolated TPU compiler subprocesses."""
import os
from pathlib import Path
import subprocess
import sys


COMMIT_SHA = '50bdcaf3735d73c926d87786c13db749d5641037'
CHECKOUT = Path('/tmp/TPUBeamSearch-selector-probe')
OUTPUT = Path('/kaggle/working/selector_probe')


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
    command = (sys.executable, '-m', 'benchmarks.beam_selector_probe', '--output', str(OUTPUT))
    with (OUTPUT / 'probe.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=CHECKOUT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


if __name__ == '__main__':
    main()
