"""Deterministically build the exact-inference Artgor Kaggle notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "third_party"
    / "artgor_cube555_v344319112"
    / "cayleypy-cube555-tpu-beam-q.ipynb"
)
OUTPUT = FOLDER / "cayleypy-cube555-tpu-beam-q-exact.ipynb"
BUILD_MANIFEST = FOLDER / "build_manifest.json"
METADATA = FOLDER / "kernel-metadata.json"

SOURCE_VERSION = 344319112
SOURCE_URL = (
    "https://www.kaggle.com/code/artgor/cayleypy-cube555-tpu-beam-q/"
    "notebook?scriptVersionId=344319112"
)
SOURCE_NOTEBOOK_SHA256 = (
    "c74613a9fa400b391aca49bb128a2f6d3b0465e8e7cb933abc9b126a317e0e0b"
)
PUBLIC_SLUG = "trydotatwo/cayleypy-cube555-tpu-beam-q-exact"
CODE_DATASET = "trydotatwo/tpu-beam-search-exact-artgor-code"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one source occurrence, found {count}: {old[:80]!r}"
        )
    return source.replace(old, new, 1)


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _validate_commit(source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a lowercase 40-hex Git SHA")
    return source_commit


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def build_notebook(source_commit: str) -> bytes:
    source_commit = _validate_commit(source_commit)
    source_bytes = SOURCE.read_bytes()
    actual_hash = _sha256(source_bytes)
    if actual_hash != SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError(
            f"frozen Artgor notebook hash changed: {actual_hash}"
        )
    notebook = json.loads(source_bytes)
    cells = notebook["cells"]
    if len(cells) != 6:
        raise RuntimeError("frozen Artgor notebook must contain six cells")

    attribution = f"""# Exact accelerated derivative

This notebook is an attributed derivative of Andrey Lukyanenko/Artgor's
[immutable Kaggle notebook]({SOURCE_URL}).  Its puzzle logic, move order,
routing, deduplication, history, endgame, and packed backpointers are preserved.

The selected split Q forward measured about **1.618x inference speedup** against
the original JAX forward at 32,768 states per TPU core on the same v5e-8 runtime.
That is an inference result, **not a claimed whole-solver speedup**.  This notebook
prints separate depth/solve timings so the wider claim is made only from a paired
end-to-end run.

Source: scriptVersionId=344319112.  TPUBeamSearch code commit:
`{source_commit}`.

---

"""
    cells[0]["source"] = attribution + _cell_source(cells[0])

    cell1 = _cell_source(cells[1])
    cell1 = _replace_once(
        cell1,
        "import json, sys, time, csv\n",
        "import json, sys, time, csv, hashlib\n",
    )
    cell1 += f"""

ARTGOR_SOURCE = {{
    "script_version_id": {SOURCE_VERSION},
    "url": "{SOURCE_URL}",
    "notebook_sha256": "{SOURCE_NOTEBOOK_SHA256}",
}}
TPU_BEAM_SEARCH_SOURCE_COMMIT = "{source_commit}"
print("Artgor source:", json.dumps(ARTGOR_SOURCE, sort_keys=True))
print("TPUBeamSearch source:", TPU_BEAM_SEARCH_SOURCE_COMMIT)
"""
    cells[1]["source"] = cell1

    cell2 = _cell_source(cells[2])
    cell2 = _replace_once(
        cell2,
        "# ---------------- configuration ----------------\n",
        """# ---------------- configuration ----------------
# exact_split is enabled only for the measured single-checkpoint Q-only path.
# Blend/QV settings fall back explicitly to the original JAX solver.
INFERENCE_ENGINE = "exact_split"
INFERENCE_CHUNK = 32768

""",
    )
    cell2 = _replace_once(
        cell2,
        "INTERNAL_BS = 16384                # model forward chunk; must divide B_LOCAL\n",
        "INTERNAL_BS = INFERENCE_CHUNK      # exact measured local Q chunk\n",
    )
    cell2 = _replace_once(
        cell2,
        "PARENT_CHUNK = None if B_GLOBAL <= 4 * 1024 * 1024 else 131072\n",
        "PARENT_CHUNK = 131072              # unchanged search selection window\n",
    )
    cells[2]["source"] = cell2

    code_mount = f"""
# ---------------- exact TPUBeamSearch code mount ----------------
code_root = None
for cand in [Path("/kaggle/input/tpu-beam-search-exact-artgor-code"),
             Path("/kaggle/input/datasets/trydotatwo/tpu-beam-search-exact-artgor-code")]:
    if cand.exists():
        code_root = cand
        break
if code_root is None:
    raise SystemExit("attach the {CODE_DATASET} dataset")

release_manifest = json.loads(
    (code_root / "release_manifest.json").read_text(encoding="utf-8")
)
if release_manifest["source_commit"] != TPU_BEAM_SEARCH_SOURCE_COMMIT:
    raise SystemExit(
        "code dataset commit mismatch: "
        f"{{release_manifest['source_commit']}} != {{TPU_BEAM_SEARCH_SOURCE_COMMIT}}"
    )
for relative, expected in release_manifest["files"].items():
    payload = (code_root / relative).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise SystemExit(f"code dataset sha256 mismatch: {{relative}}")
sys.path.insert(0, str(code_root))

from tpu_beam_search import (
    ArtgorExactConfig,
    beam_solve_v_only_spmd_packed_exact,
    choose_artgor_inference_engine,
    prepare_artgor_exact_beam_runtime,
)
from jax.sharding import Mesh

"""
    cell3 = code_mount + _cell_source(cells[3])
    cell3 = _replace_once(
        cell3,
        "mesh = make_mesh(devices)\n",
        """mesh = make_mesh(devices)  # original JAX fallback uses axis 'cores'
exact_mesh = Mesh(np.asarray(devices), ("core",))
engine_decision = choose_artgor_inference_engine(
    INFERENCE_ENGINE, BLEND_CHECKPOINTS, QV_CONSISTENCY,
)
EXACT_CONFIG = ArtgorExactConfig(
    prefix_bm=4096, head_bm=256, head_bk=1024, head_bn=128,
    dense_rounding="late", inference_chunk=INFERENCE_CHUNK,
    parent_chunk=PARENT_CHUNK,
)
EXACT_RUNTIME = None
solver_mesh = mesh
if engine_decision.selected == "exact_split":
    EXACT_RUNTIME = prepare_artgor_exact_beam_runtime(
        v_params, mesh=exact_mesh, exact_config=EXACT_CONFIG,
        state_storage_len=STATE_SIZE,
    )
    solver_mesh = exact_mesh
print(f"inference engine: {engine_decision.selected} ({engine_decision.reason})")
""",
    )
    cells[3]["source"] = cell3

    cell4 = _cell_source(cells[4])
    cell4 = _replace_once(
        cell4,
        "# ---------------- frame transforms (mirror cube555/scripts/30_solve.py) ----------\n",
        """# ---------------- solver selection ----------------
if engine_decision.selected == "exact_split":
    beam_solver = beam_solve_v_only_spmd_packed_exact
    beam_solver_extra = {
        "exact_config": EXACT_CONFIG,
        "exact_runtime": EXACT_RUNTIME,
    }
else:
    beam_solver = beam_solve_v_only_spmd_packed
    beam_solver_extra = {}

# ---------------- frame transforms (mirror cube555/scripts/30_solve.py) ----------
""",
    )
    cell4 = _replace_once(
        cell4,
        "r = beam_solve_v_only_spmd_packed(\n",
        "r = beam_solver(\n",
    )
    cell4 = _replace_once(
        cell4,
        "list(u), v_params, all_moves, V0, hash_vec, mesh,\n",
        "list(u), v_params, all_moves, V0, hash_vec, solver_mesh,\n",
    )
    cell4 = _replace_once(
        cell4,
        "q_mode=Q_MODE, qv_consistency=QV_CONSISTENCY,\n            )\n",
        """q_mode=Q_MODE, qv_consistency=QV_CONSISTENCY,
                **beam_solver_extra,
            )
""",
    )
    cells[4]["source"] = cell4

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    notebook["metadata"].setdefault("tpu_beam_search", {})
    notebook["metadata"]["tpu_beam_search"] = {
        "artgor_script_version": SOURCE_VERSION,
        "source_commit": source_commit,
        "engine": "exact_split",
    }
    return (
        json.dumps(
            notebook,
            ensure_ascii=False,
            sort_keys=True,
            indent=1,
        )
        + "\n"
    ).encode("utf-8")


def _metadata_bytes() -> bytes:
    metadata = {
        "id": PUBLIC_SLUG,
        "title": "CayleyPy Cube555 TPU Beam Q - Exact 1.6x Inference",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "dataset_sources": [
            "artgor/cube555-tpu-artifacts",
            CODE_DATASET,
        ],
        "competition_sources": ["cayley-py-555-cube"],
        "kernel_sources": [],
        "model_sources": [],
    }
    return (
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _manifest_bytes(
    notebook_bytes: bytes, source_commit: str
) -> bytes:
    manifest = {
        "artgor_script_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "source_commit": source_commit,
        "builder_sha256": _sha256(Path(__file__).read_bytes()),
        "generated_notebook_sha256": _sha256(notebook_bytes),
    }
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _write_or_check(path: Path, expected: bytes, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != expected:
            raise SystemExit(f"generated artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    source_commit = _validate_commit(args.source_commit or _current_commit())
    notebook_bytes = build_notebook(source_commit)
    _write_or_check(OUTPUT, notebook_bytes, check=args.check)
    _write_or_check(METADATA, _metadata_bytes(), check=args.check)
    _write_or_check(
        BUILD_MANIFEST,
        _manifest_bytes(notebook_bytes, source_commit),
        check=args.check,
    )
    action = "verified" if args.check else "built"
    print(f"{action}: {OUTPUT}")


if __name__ == "__main__":
    main()
