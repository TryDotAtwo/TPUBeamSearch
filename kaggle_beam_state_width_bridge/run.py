"""Private eight-TPU compact-state/final-materializer physical gate."""
import os
from pathlib import Path
import subprocess
import sys


COMMIT_SHA = "6756730b2436b86b62676ae0723f496599aa2fc5"
CHECKOUT = Path("/tmp/TPUBeamSearch-state-width-bridge")
OUTPUT = Path("/kaggle/working/state_width_bridge_execution")


def main():
    subprocess.run((sys.executable, "-m", "pip", "install", "--quiet",
                    "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1"), check=True)
    subprocess.run(("git", "clone", "https://github.com/TryDotAtwo/TPUBeamSearch.git",
                    str(CHECKOUT)), check=True)
    subprocess.run(("git", "checkout", "--detach", COMMIT_SHA), cwd=CHECKOUT, check=True)
    actual = subprocess.check_output(("git", "rev-parse", "HEAD"),
                                     cwd=CHECKOUT, text=True).strip()
    dirty = subprocess.check_output(("git", "status", "--porcelain"),
                                    cwd=CHECKOUT, text=True).strip()
    if actual != COMMIT_SHA or dirty:
        raise RuntimeError("pinned source checkout is not clean")
    env = os.environ.copy()
    env.update(JAX_ENABLE_X64="False", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join((str(CHECKOUT), str(CHECKOUT / "src"))))
    subprocess.run((sys.executable, "-m", "benchmarks.beam_state_width_bridge_execution",
                    "--output", str(OUTPUT)), cwd=CHECKOUT, env=env, check=True)


if __name__ == "__main__":
    main()
