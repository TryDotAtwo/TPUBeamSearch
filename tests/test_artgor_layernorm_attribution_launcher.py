import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_layernorm_attribution"


def test_launcher_is_private_tpu_and_pinned_to_full_sha():
    metadata = json.loads((FOLDER / "kernel-metadata.json").read_text())
    assert (
        metadata["id"]
        == "trydotatwo/tpu-artgor-layernorm-arithmetic-attribution"
    )
    assert metadata["enable_tpu"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["is_private"] is True
    source = (FOLDER / metadata["code_file"]).read_text()
    match = re.search(r'^COMMIT_SHA = "([0-9a-f]{40})"$', source, re.MULTILINE)
    assert match
    assert "benchmarks.artgor_layernorm_attribution" in source
    assert "jax[tpu]==0.10.2" in source
