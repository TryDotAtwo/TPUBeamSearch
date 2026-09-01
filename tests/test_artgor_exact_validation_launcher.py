import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "kaggle_artgor_exact_validation"
EXPECTED_COMMIT = "403fae385b909d7fa091294479b8ec2525df23fe"


def test_launcher_is_source_pinned_and_streams_a_persistent_log():
    path = FOLDER / "run_validation.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert f'COMMIT_SHA = "{EXPECTED_COMMIT}"' in source
    assert re.search(r'COMMIT_SHA = "[0-9a-f]{40}"', source)
    assert "https://github.com/TryDotAtwo/TPUBeamSearch.git" in source
    assert '("git", "checkout", "--detach", COMMIT_SHA)' in source
    assert "--detach" in source
    assert "rev-parse" in source
    assert "jax[tpu]==0.10.2" in source
    assert "jaxlib==0.10.2" in source
    assert "libtpu==0.0.42.1" in source
    assert "JAX_ENABLE_X64" in source
    assert "benchmarks.artgor_exact_notebook_validation" in source
    assert "/kaggle/working/artgor_exact_notebook_validation" in source
    assert "validation.log" in source
    assert "stdout=subprocess.PIPE" in source
    assert "log.flush()" in source


def test_launcher_metadata_is_private_single_tpu_job_with_assets():
    metadata = json.loads(
        (FOLDER / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["id"] == (
        "trydotatwo/tpu-artgor-exact-notebook-validation"
    )
    assert metadata["code_file"] == "run_validation.py"
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] is True
    assert metadata["enable_tpu"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == [
        "artgor/cube555-tpu-artifacts"
    ]
    assert metadata["competition_sources"] == ["cayley-py-555-cube"]
