"""The final-residual A/B launcher pins one published source revision."""
import ast
import json
from pathlib import Path
import re


def test_final_residual_launcher_is_private_tpu_and_reproducibly_pinned():
    folder = Path(__file__).resolve().parents[1] / "kaggle_final_residual_ab"
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "trydotatwo/tpu-final-residual-ab"
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
    assert assignments["COMMIT_SHA"] == "267df37cd3a35b19ad6250d43768bfd5b536b67c"
    assert re.fullmatch(r"[0-9a-f]{40}", assignments["COMMIT_SHA"])
    assert assignments["REPOSITORY"] == "https://github.com/TryDotAtwo/TPUBeamSearch.git"
    strings = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1",
        "benchmarks.stream1_final_residual_ab", "benchmark.log", "--detach",
        "PYTHONPATH", "PYTHONUNBUFFERED",
        "/kaggle/working/final_residual_ab",
    } <= strings
