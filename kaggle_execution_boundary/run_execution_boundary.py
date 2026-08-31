"""Private execution-boundary bundle; fetch source from one published Git SHA."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

COMMIT_SHA = "45062324d368f4849adb6d572d21d54f75854d79"
REPOSITORY = "https://github.com/TryDotAtwo/TPUBeamSearch.git"
CHECKOUT = Path("/tmp/TPUBeamSearch")
OUTPUT = Path("/kaggle/working/execution_boundary")


def main():
    if not re.fullmatch(r"[0-9a-f]{40}", COMMIT_SHA):
        raise ValueError("runner requires a full published source commit SHA")
    subprocess.run((sys.executable, "-m", "pip", "install", "--quiet",
                    "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1"), check=True)
    subprocess.run(("git", "clone", REPOSITORY, str(CHECKOUT)), check=True)
    subprocess.run(("git", "checkout", "--detach", COMMIT_SHA), cwd=CHECKOUT, check=True)
    actual = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=CHECKOUT, text=True).strip()
    if actual != COMMIT_SHA:
        raise RuntimeError(f"expected {COMMIT_SHA}, got {actual}")
    print(f"SOURCE_COMMIT={actual}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(CHECKOUT), str(CHECKOUT / "src")))
    environment["PYTHONUNBUFFERED"] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = (sys.executable, "-m", "benchmarks.stream1_execution_boundary", "--output", str(OUTPUT))
    with (OUTPUT / "benchmark.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=CHECKOUT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


if __name__ == "__main__":
    main()
