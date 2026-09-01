import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks" / "artgor_cube555_exact_tpu"
NOTEBOOK = FOLDER / "cayleypy-cube555-tpu-beam-q-exact.ipynb"


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
