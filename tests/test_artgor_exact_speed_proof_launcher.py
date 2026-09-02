import ast
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_exact_speed_proof"


def test_speed_proof_launcher_is_pinned_and_writes_a_persistent_log():
    launcher = FOLDER / "run_speed_proof.py"
    assert launcher.is_file(), "speed-proof launcher is not implemented"
    source = launcher.read_text(encoding="utf-8")
    ast.parse(source)
    match = re.search(r'COMMIT_SHA = "([0-9a-f]{40})"', source)
    assert match
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{match.group(1)}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{match.group(1)}:benchmarks/artgor_exact_speed_proof.py",
        ],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    assert "https://github.com/TryDotAtwo/TPUBeamSearch.git" in source
    assert "jax[tpu]==0.10.2" in source
    assert "jaxlib==0.10.2" in source
    assert "libtpu==0.0.42.1" in source
    assert "benchmarks.artgor_exact_speed_proof" in source
    assert "/kaggle/working/artgor_exact_speed_proof" in source
    assert "speed_proof.log" in source


def test_speed_proof_metadata_uses_one_private_tpu_session_without_competition():
    metadata_path = FOLDER / "kernel-metadata.json"
    assert metadata_path.is_file(), "speed-proof metadata is not implemented"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "trydotatwo/tpu-artgor-exact-speed-proof"
    assert metadata["code_file"] == "run_speed_proof.py"
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] is True
    assert metadata["enable_tpu"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == ["artgor/cube555-tpu-artifacts"]
    assert metadata["competition_sources"] == []
