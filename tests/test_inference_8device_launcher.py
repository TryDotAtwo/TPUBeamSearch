"""The private launcher must fetch exactly one public source revision."""
import ast
import json
from pathlib import Path
import re


def test_inference_launcher_is_private_tpu_and_pins_the_public_repository():
    folder = Path(__file__).resolve().parents[1] / "kaggle_inference_8device"
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "trydotatwo/tpu-exact-inference-8-device"
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
    assert re.fullmatch(r"[0-9a-f]{40}", assignments["COMMIT_SHA"])
    assert assignments["REPOSITORY"] == "https://github.com/TryDotAtwo/TPUBeamSearch.git"
    strings = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "jax[tpu]==0.10.2", "jaxlib==0.10.2", "libtpu==0.0.42.1",
        "benchmarks.stream1_inference_8device", "benchmark.log", "--detach",
        "PYTHONPATH", "PYTHONUNBUFFERED", "/kaggle/working/inference_8device",
    } <= strings
