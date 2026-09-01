"""The exact-frontier Kaggle launcher is private, TPU-only, and source-pinned."""

import ast
import json
from pathlib import Path
import re


def test_exact_frontier_launcher_contract():
    folder = Path(__file__).resolve().parents[1] / "kaggle_exact_inference_frontier"
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "trydotatwo/tpu-exact-inference-frontier"
    assert metadata["is_private"] is True
    assert metadata["enable_tpu"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == ["artgor/cube555-tpu-artifacts"]

    source = (folder / metadata["code_file"]).read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert assignments["COMMIT_SHA"] == "fc5c87ae5c49c0a92d4ccd634831e8980a7f44e8"
    assert re.fullmatch(r"[0-9a-f]{40}", assignments["COMMIT_SHA"])
    assert assignments["REPOSITORY"] == "https://github.com/TryDotAtwo/TPUBeamSearch.git"
    strings = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1",
        "benchmarks.stream1_exact_inference_frontier", "benchmark.log",
        "--detach", "PYTHONPATH", "PYTHONUNBUFFERED",
        "/kaggle/working/exact_inference_frontier",
    } <= strings
