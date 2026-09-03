"""Run the composed prefix gate from an immutable public source commit."""
import os
from pathlib import Path
import re
import subprocess
import sys

COMMIT_SHA = '4ab805bb2bac756212dfd882cb23f7e0ba3a601f'
REPOSITORY = 'https://github.com/TryDotAtwo/TPUBeamSearch.git'
CHECKOUT = Path('/tmp/TPUBeamSearch-artgor-prefix-gate')
OUTPUT = Path('/kaggle/working/artgor_invstd_capture')


def main():
    if not re.fullmatch(r'[0-9a-f]{40}', COMMIT_SHA):
        raise ValueError('requires a full published source SHA')
    subprocess.run((sys.executable,'-m','pip','install','--quiet','jax[tpu]==0.10.2','jaxlib==0.10.2','libtpu==0.0.42.1'),check=True)
    subprocess.run(('git','clone',REPOSITORY,str(CHECKOUT)),check=True)
    subprocess.run(('git','checkout','--detach',COMMIT_SHA),cwd=CHECKOUT,check=True)
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64='True',XLA_PYTHON_CLIENT_MEM_FRACTION='0.95',
               PYTHONPATH=os.pathsep.join((str(CHECKOUT),str(CHECKOUT/'src'))),PYTHONUNBUFFERED='1')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    command = (sys.executable,'-m','benchmarks.artgor_prefix_capture','--output',str(OUTPUT),'--include-invstd')
    with (OUTPUT/'prefix_gate.log').open('w',encoding='utf-8') as log:
        with subprocess.Popen(command,cwd=CHECKOUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code,command)


if __name__ == '__main__':
    main()
