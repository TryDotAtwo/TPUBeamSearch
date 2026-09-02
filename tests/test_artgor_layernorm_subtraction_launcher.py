import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_layernorm_subtraction"


def test_launcher_is_private_tpu_and_pinned():
    metadata = json.loads((FOLDER / "kernel-metadata.json").read_text())
    assert metadata["id"] == "trydotatwo/tpu-artgor-layernorm-subtraction"
    assert metadata["enable_tpu"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["is_private"] is True
    source = (FOLDER / metadata["code_file"]).read_text()
    assert re.search(r'^COMMIT_SHA = "[0-9a-f]{40}"$', source, re.MULTILINE)
    assert "benchmarks.artgor_layernorm_subtraction" in source
    assert "jax[tpu]==0.10.2" in source
