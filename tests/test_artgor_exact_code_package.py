import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from kaggle_artgor_exact_code.build_release import (
    RUNTIME_FILES,
    build_release,
)


ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def test_code_release_is_allowlisted_hashed_and_commit_pinned(tmp_path):
    source_commit = _head()
    output = build_release(tmp_path / "release", source_commit=source_commit)
    manifest = json.loads(
        (output / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_commit"] == source_commit
    assert manifest["source_repository"] == (
        "https://github.com/TryDotAtwo/TPUBeamSearch.git"
    )
    assert manifest["artgor_script_version"] == 344319112
    assert set(manifest["files"]) == set(RUNTIME_FILES)
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == (
            expected
        )
    assert not any(path.suffix == ".pyc" for path in output.rglob("*"))


def test_release_uses_bytes_from_the_named_commit_not_the_worktree(tmp_path):
    source_commit = _head()
    output = build_release(tmp_path / "release", source_commit=source_commit)
    expected = subprocess.check_output(
        ["git", "show", f"{source_commit}:src/tpu_beam_search/__init__.py"],
        cwd=ROOT,
    )
    assert (output / "tpu_beam_search" / "__init__.py").read_bytes() == expected


def test_release_rejects_stale_output_and_non_commit_sha(tmp_path):
    output = tmp_path / "release"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        build_release(output, source_commit=_head())
    with pytest.raises(ValueError, match="40-hex"):
        build_release(tmp_path / "new", source_commit="main")


def test_dataset_metadata_names_only_tpu_beam_search_code(tmp_path):
    output = build_release(tmp_path / "release", source_commit=_head())
    metadata = json.loads(
        (output / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["id"] == (
        "trydotatwo/tpu-beam-search-exact-artgor-code"
    )
    assert metadata["licenses"] == [{"name": "MIT"}]
