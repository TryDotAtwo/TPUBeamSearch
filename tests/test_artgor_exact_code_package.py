import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

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
    with zipfile.ZipFile(output / "tpu_beam_search_runtime.zip") as runtime_zip:
        for relative, expected in manifest["files"].items():
            assert hashlib.sha256(runtime_zip.read(relative)).hexdigest() == expected
        assert not any(name.endswith(".pyc") for name in runtime_zip.namelist())


def test_release_uses_bytes_from_the_named_commit_not_the_worktree(tmp_path):
    source_commit = _head()
    output = build_release(tmp_path / "release", source_commit=source_commit)
    expected = subprocess.check_output(
        ["git", "show", f"{source_commit}:src/tpu_beam_search/__init__.py"],
        cwd=ROOT,
    )
    with zipfile.ZipFile(output / "tpu_beam_search_runtime.zip") as runtime_zip:
        assert runtime_zip.read("tpu_beam_search/__init__.py") == expected


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


def test_release_is_flat_and_runtime_zip_matches_manifest(tmp_path):
    output = build_release(tmp_path / "release", source_commit=_head())
    manifest = json.loads(
        (output / "release_manifest.json").read_text(encoding="utf-8")
    )
    archive = output / "tpu_beam_search_runtime.zip"

    assert archive.is_file()
    assert not any(path.is_dir() for path in output.iterdir())
    assert manifest["runtime_archive"] == {
        "name": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    with zipfile.ZipFile(archive) as runtime_zip:
        assert set(runtime_zip.namelist()) == set(RUNTIME_FILES)
        for relative, expected in manifest["files"].items():
            assert hashlib.sha256(runtime_zip.read(relative)).hexdigest() == expected
