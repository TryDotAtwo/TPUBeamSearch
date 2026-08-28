from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


COMMIT_SHA = "36be532d8a18bcdf073de70a3e175b41033d2027"
REPOSITORY = "https://github.com/TryDotAtwo/TPUBeamSearch.git"
CHECKOUT = Path("/tmp/TPUBeamSearch")


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


run(sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "libtpu")
run("git", "clone", REPOSITORY, str(CHECKOUT))
run("git", "checkout", "--detach", COMMIT_SHA, cwd=CHECKOUT)
actual_sha = subprocess.check_output(
    ("git", "rev-parse", "HEAD"), cwd=CHECKOUT, text=True
).strip()
if actual_sha != COMMIT_SHA:
    raise RuntimeError(f"expected {COMMIT_SHA}, checked out {actual_sha}")

environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join(
    (str(CHECKOUT), str(CHECKOUT / "src"), str(CHECKOUT / "benchmarks"))
)
subprocess.run(
    (sys.executable, "-m", "benchmarks.stream1_layernorm_full_mlp"),
    cwd=CHECKOUT,
    env=environment,
    check=True,
)
