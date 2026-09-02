import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_layernorm_invstd"


def test_launcher_is_private_tpu_and_pinned():
    metadata = json.loads((FOLDER / "kernel-metadata.json").read_text())
    assert metadata["id"] == "trydotatwo/tpu-artgor-layernorm-invstd"
    assert metadata["enable_tpu"] and not metadata["enable_gpu"]
    assert metadata["is_private"]
    source = (FOLDER / metadata["code_file"]).read_text()
    assert re.search(r'^COMMIT_SHA = "[0-9a-f]{40}"$', source, re.MULTILINE)
    assert "benchmarks.artgor_layernorm_invstd" in source
