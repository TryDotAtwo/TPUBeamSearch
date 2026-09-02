import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_layernorm_boundary_replay"


def test_boundary_replay_launcher_is_private_and_pinned_to_public_source():
    metadata = json.loads((FOLDER / "kernel-metadata.json").read_text())
    source = (FOLDER / "run_boundary_replay.py").read_text()
    assert metadata["id"] == "trydotatwo/tpu-artgor-layernorm-boundary-replay"
    assert metadata["is_private"] is True
    assert metadata["enable_tpu"] is True
    assert "benchmarks.artgor_layernorm_boundary_replay" in source
    assert "EXPECTED_SOURCE_COMMIT" in source
