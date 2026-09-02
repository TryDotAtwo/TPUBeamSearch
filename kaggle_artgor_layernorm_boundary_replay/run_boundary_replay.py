"""Run the LayerNorm boundary replay from one public source SHA."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


EXPECTED_SOURCE_COMMIT = "0000000000000000000000000000000000000000"
REPOSITORY = "https://github.com/TryDotAtwo/TPUBeamSearch.git"
CHECKOUT = Path("/tmp/TPUBeamSearch-artgor-layernorm-boundary-replay")
OUTPUT = Path("/kaggle/working/artgor_layernorm_boundary_replay")


def main() -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_SOURCE_COMMIT):
        raise ValueError("runner requires a full published source commit SHA")
    if CHECKOUT.exists():
        raise RuntimeError(f"refusing a stale checkout: {CHECKOUT}")
    subprocess.run((
        sys.executable, "-m", "pip", "install", "--quiet",
        "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1",
    ), check=True)
    subprocess.run(("git", "clone", REPOSITORY, str(CHECKOUT)), check=True)
    subprocess.run(
        ("git", "checkout", "--detach", EXPECTED_SOURCE_COMMIT),
        cwd=CHECKOUT, check=True,
    )
    actual = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=CHECKOUT, text=True,
    ).strip()
    if actual != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError(f"expected {EXPECTED_SOURCE_COMMIT}, got {actual}")
    print(f"SOURCE_COMMIT={actual}", flush=True)
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "True"
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    environment["PYTHONPATH"] = os.pathsep.join((str(CHECKOUT), str(CHECKOUT / "src")))
    environment["PYTHONUNBUFFERED"] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable, "-m", "benchmarks.artgor_layernorm_boundary_replay",
        "--output", str(OUTPUT),
    )
    with (OUTPUT / "boundary_replay.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=CHECKOUT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
