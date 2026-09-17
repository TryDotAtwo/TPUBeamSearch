"""Private physical execution gate for split_gather."""
import os
import subprocess
import sys
from pathlib import Path

COMMIT_SHA = "a0450117eaeff1d4311725eb6fb19d5aeed96f7c"
CHECKOUT = Path("/tmp/TPUBeamSearch-split-execution")
OUTPUT = Path("/kaggle/working/split_gather_execution")


def main():
    subprocess.run((sys.executable, "-m", "pip", "install", "--quiet",
                    "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1"), check=True)
    subprocess.run(("git", "clone", "https://github.com/TryDotAtwo/TPUBeamSearch.git", str(CHECKOUT)), check=True)
    subprocess.run(("git", "checkout", "--detach", COMMIT_SHA), cwd=CHECKOUT, check=True)
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64="False", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join((str(CHECKOUT), str(CHECKOUT / "src"))))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    subprocess.run((sys.executable, "-m", "benchmarks.beam_split_gather_execution",
                    "--output", str(OUTPUT)), cwd=CHECKOUT, env=env, check=True)


if __name__ == "__main__":
    main()
