import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_layernorm_monolithic_match"


def test_monolithic_match_launcher_is_private_and_pinned():
    metadata = json.loads((FOLDER / "kernel-metadata.json").read_text())
    source = (FOLDER / "run_monolithic_match.py").read_text()
    assert metadata["id"] == "trydotatwo/tpu-artgor-layernorm-monolithic-match"
    assert metadata["is_private"] is True
    assert metadata["enable_tpu"] is True
    assert "benchmarks.artgor_layernorm_monolithic_match" in source
    assert "EXPECTED_SOURCE_COMMIT" in source
