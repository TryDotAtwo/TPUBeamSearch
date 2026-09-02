import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import pytest

from kaggle_artgor_exact_code.build_release import build_release


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks" / "artgor_cube555_exact_tpu"
NOTEBOOK = FOLDER / "cayleypy-cube555-tpu-beam-q-exact.ipynb"


def _mount_preamble() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    start = source.index("# ---------------- exact TPUBeamSearch code mount")
    stop = source.index("from tpu_beam_search import")
    return source[start:stop]


def _runtime_commit() -> str:
    return (FOLDER / "runtime-source-commit.txt").read_text(
        encoding="utf-8"
    ).strip()


def test_generated_notebook_preserves_six_cell_flow_and_exact_engine():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 6
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        "markdown",
        "code",
        "code",
        "code",
        "code",
        "code",
    ]
    sources = [
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    ]
    for source in sources[1:]:
        ast.parse(source)
    joined = "\n".join(sources)
    assert 'INFERENCE_ENGINE = "exact_split"' in joined
    assert "INFERENCE_CHUNK = 32768" in joined
    assert "beam_solve_v_only_spmd_packed_exact" in joined
    assert "prepare_artgor_exact_beam_runtime" in joined
    assert "choose_artgor_inference_engine" in joined
    assert "QV_CONSISTENCY" in joined and "BLEND_CHECKPOINTS" in joined
    assert "scriptVersionId=344319112" in joined
    assert "inference speedup" in joined.lower()
    assert "whole-solver speedup" in joined.lower()
    assert "release_manifest.json" in joined
    assert "sha256" in joined
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None


def test_generated_notebook_records_its_frozen_source_and_builder_hash():
    manifest = json.loads(
        (FOLDER / "build_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artgor_script_version"] == 344319112
    assert manifest["source_notebook_sha256"] == (
        "c74613a9fa400b391aca49bb128a2f6d3b0465e8e7cb933abc9b126a317e0e0b"
    )
    assert manifest["generated_notebook_sha256"] == hashlib.sha256(
        NOTEBOOK.read_bytes()
    ).hexdigest()
    assert manifest["source_commit"] == (
        FOLDER / "runtime-source-commit.txt"
    ).read_text(encoding="utf-8").strip()


def test_kaggle_metadata_is_private_tpu_only_and_attaches_both_sources():
    metadata = json.loads(
        (FOLDER / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["id"] == (
        "trydotatwo/cayleypy-cube555-tpu-beam-q-exact"
    )
    assert metadata["code_file"] == NOTEBOOK.name
    assert metadata["kernel_type"] == "notebook"
    assert metadata["is_private"] is True
    assert metadata["enable_tpu"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert set(metadata["dataset_sources"]) == {
        "artgor/cube555-tpu-artifacts",
        "trydotatwo/tpu-beam-search-exact-artgor-code",
    }
    assert metadata["competition_sources"] == ["cayley-py-555-cube"]


def test_kaggle_title_resolves_to_declared_kernel_slug():
    metadata = json.loads(
        (FOLDER / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    resolved = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    assert resolved == metadata["id"].split("/", 1)[1]


def test_generated_notebook_mounts_verified_flat_runtime_zip(tmp_path, monkeypatch):
    release = build_release(
        tmp_path / "release", source_commit=_runtime_commit()
    )
    monkeypatch.setenv("TPU_BEAM_SEARCH_CODE_ROOT", str(release))
    old_path = list(sys.path)
    scope = {
        "Path": Path,
        "TPU_BEAM_SEARCH_SOURCE_COMMIT": _runtime_commit(),
        "json": json,
        "os": os,
        "sys": sys,
    }
    try:
        exec(_mount_preamble(), scope)
        archive = release / "tpu_beam_search_runtime.zip"
        assert scope["runtime_archive"] == archive
        assert sys.path[0] == str(archive)
    finally:
        sys.path[:] = old_path


def test_generated_notebook_rejects_tampered_runtime_zip(tmp_path, monkeypatch):
    release = build_release(
        tmp_path / "release", source_commit=_runtime_commit()
    )
    archive = release / "tpu_beam_search_runtime.zip"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    monkeypatch.setenv("TPU_BEAM_SEARCH_CODE_ROOT", str(release))
    scope = {
        "Path": Path,
        "TPU_BEAM_SEARCH_SOURCE_COMMIT": _runtime_commit(),
        "json": json,
        "os": os,
        "sys": sys,
    }
    with pytest.raises(SystemExit, match="archive sha256 mismatch"):
        exec(_mount_preamble(), scope)


def test_generated_notebook_mounts_kaggle_extracted_runtime(tmp_path, monkeypatch):
    release = build_release(
        tmp_path / "release", source_commit=_runtime_commit()
    )
    archive = release / "tpu_beam_search_runtime.zip"
    extracted = release / archive.stem
    with zipfile.ZipFile(archive) as runtime_zip:
        runtime_zip.extractall(extracted)
    archive.unlink()

    monkeypatch.setenv("TPU_BEAM_SEARCH_CODE_ROOT", str(release))
    old_path = list(sys.path)
    scope = {
        "Path": Path,
        "TPU_BEAM_SEARCH_SOURCE_COMMIT": _runtime_commit(),
        "json": json,
        "os": os,
        "sys": sys,
    }
    try:
        exec(_mount_preamble(), scope)
        assert scope["runtime_import_root"] == extracted
        assert sys.path[0] == str(extracted)
    finally:
        sys.path[:] = old_path
