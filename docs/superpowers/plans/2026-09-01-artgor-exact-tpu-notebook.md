# Artgor Exact Accelerated TPU Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and validate a copy-and-run Kaggle TPU notebook based on Artgor script version `344319112`, preserving its search behavior while using the bitwise-exact split inference engine measured at approximately `1.6x` the original JAX forward.

**Architecture:** Freeze and hash the upstream six-cell notebook and its two Python modules, expose the selected prefix/Pallas-head pair as a stable two-dispatch API, and add a staged beam-depth executor that assembles 64 exact 32K Q chunks into a device-resident 120 MiB-per-core Q tensor before one unchanged 131,072-parent streaming search scan. Build the public notebook and code dataset deterministically from the Git checkout, then use one source-pinned private Kaggle TPU job to gate exact inference, exact one-depth search, a short A/B, and a real-width replay-valid solve before publication.

**Tech Stack:** Python 3.12, JAX/jaxlib 0.10.2, Pallas/Mosaic TPU, libtpu 0.0.42.1, NumPy, pytest, Kaggle CLI/API, GitHub.

**Spec:** `docs/superpowers/specs/2026-09-01-artgor-exact-tpu-notebook-design.md`

## Global Constraints

- Work inline in the current task; do not use subagents.
- GitHub `TryDotAtwo/TPUBeamSearch` is the source of truth; Kaggle validation clones a full pinned public Git SHA.
- Preserve Artgor notebook script version `344319112`, checkpoint `q555_2k_BEST.pt`, `uint8[150]` states, BF16 weights/outputs, 30 minimizing-Q scores, move order, history, endgame, routing, dedup, top-K, and packed-backpointer semantics.
- Selected inference constants are prefix `BM=4096`; Pallas head `BM=256`, `BK=1024`, `BN=128`, `dense_rounding="late"`; local inference chunk `32768`; parent selection window `131072`.
- Prefix and head must remain separately compiled device-resident dispatches and must never be enclosed in one outer `jax.jit`.
- Unsupported blend or nonzero QV-consistency configurations must select `original_jax` explicitly before compilation.
- Exactness means finite outputs, zero unequal BF16 elements, and identical hashes; argmin or top-K agreement alone cannot promote a candidate.
- Keep only one Kaggle TPU session active at a time.
- Do not modify the existing BN path or stage/delete unrelated untracked artifacts.
- Do not claim a whole-solver speedup from inference-only measurements.

---

### Task 1: Freeze and verify the Artgor source snapshot

**Files:**
- Create: `third_party/artgor_cube555_v344319112/PROVENANCE.md`
- Create: `third_party/artgor_cube555_v344319112/manifest.json`
- Create: `third_party/artgor_cube555_v344319112/cayleypy-cube555-tpu-beam-q.ipynb`
- Create: `third_party/artgor_cube555_v344319112/jax_model.py`
- Create: `third_party/artgor_cube555_v344319112/jax_beam_spmd_v_only.py`
- Create: `tests/test_artgor_source_snapshot.py`

**Interfaces:**
- Consumes: the already downloaded immutable source under `test_results/artgor_cube555_tpu/`.
- Produces: `manifest.json` with `script_version_id`, source URL, filenames, and SHA-256 values used by the notebook builder and release packager.

- [ ] **Step 1: Write the failing source-snapshot test**

```python
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "third_party" / "artgor_cube555_v344319112"


def test_artgor_snapshot_is_the_frozen_script_version():
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    assert manifest["script_version_id"] == 344319112
    assert manifest["source_url"].endswith("scriptVersionId=344319112")
    expected = {
        "cayleypy-cube555-tpu-beam-q.ipynb":
            "c74613a9fa400b391aca49bb128a2f6d3b0465e8e7cb933abc9b126a317e0e0b",
        "jax_model.py":
            "6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e",
        "jax_beam_spmd_v_only.py":
            "aaa0dbe16fd82a0f2bc08f1216f4e87c8a2a63c855f5d7012b6c18d8b57d42cb",
    }
    assert manifest["sha256"] == expected
    for name, digest in expected.items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == digest
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest tests/test_artgor_source_snapshot.py -q`

Expected: FAIL because `third_party/artgor_cube555_v344319112/manifest.json` does not exist.

- [ ] **Step 3: Add the frozen files, manifest, and attribution**

Use exact byte copies of the three downloaded source files. Write this manifest:

```json
{
  "script_version_id": 344319112,
  "source_url": "https://www.kaggle.com/code/artgor/cayleypy-cube555-tpu-beam-q/notebook?scriptVersionId=344319112",
  "sha256": {
    "cayleypy-cube555-tpu-beam-q.ipynb": "c74613a9fa400b391aca49bb128a2f6d3b0465e8e7cb933abc9b126a317e0e0b",
    "jax_model.py": "6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e",
    "jax_beam_spmd_v_only.py": "aaa0dbe16fd82a0f2bc08f1216f4e87c8a2a63c855f5d7012b6c18d8b57d42cb"
  }
}
```

`PROVENANCE.md` must name Andrey Lukyanenko/Artgor, link the immutable Kaggle version, state that the snapshot exists solely to create and validate the attributed derivative, and list the exact hashes without inventing a license.

- [ ] **Step 4: Run the source test**

Run: `python -m pytest tests/test_artgor_source_snapshot.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only the snapshot task**

```bash
git add third_party/artgor_cube555_v344319112 tests/test_artgor_source_snapshot.py
git commit -m "Freeze Artgor TPU notebook source"
```

---

### Task 2: Promote the selected exact inference engine to a stable API

**Files:**
- Create: `src/tpu_beam_search/artgor_exact_inference.py`
- Create: `tests/test_artgor_exact_inference.py`
- Modify: `src/tpu_beam_search/__init__.py`

**Interfaces:**
- Consumes: `prepare_exact_layernorm_inference_weights`, `stream1_layernorm_exact_prefix`, `pallas_layernorm_dense`, `make_sharded_inference`, and Artgor parameter dictionaries.
- Produces: `ArtgorExactConfig`, `ArtgorExactInference`, `prepare_artgor_exact_inference_from_weights(weights, architecture, *, mesh, config, interpret=False)`, `prepare_artgor_exact_inference(params, *, mesh, config, state_storage_len=150, interpret=False)`, and `choose_artgor_inference_engine(requested, blend_checkpoints, qv_consistency)`.

- [ ] **Step 1: Write failing configuration and numerical tests**

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from test_layernorm_followup import model_fixture
from tpu_beam_search.artgor_exact_inference import (
    ArtgorExactConfig,
    choose_artgor_inference_engine,
    prepare_artgor_exact_inference_from_weights,
)
from tpu_beam_search.stream1_layernorm_reference import (
    stream1_layernorm_reference_inference,
)


def test_selected_config_is_frozen_and_invalid_overrides_fail():
    assert ArtgorExactConfig() == ArtgorExactConfig(
        prefix_bm=4096, head_bm=256, head_bk=1024,
        head_bn=128, dense_rounding="late", inference_chunk=32768,
        parent_chunk=131072,
    )
    with np.testing.assert_raises(ValueError):
        ArtgorExactConfig(inference_chunk=24576).validate()


def test_unsupported_modes_choose_explicit_jax_fallback():
    assert choose_artgor_inference_engine("exact_split", None, 0.0).selected == "exact_split"
    decision = choose_artgor_inference_engine("exact_split", ["q555_6k.pt"], 0.0)
    assert decision.selected == "original_jax"
    assert "BLEND_CHECKPOINTS" in decision.reason


def test_interpreted_prefix_and_pallas_head_equal_reference():
    _, states, architecture, weights = model_fixture()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("core",))
    engine, prepared = prepare_artgor_exact_inference_from_weights(
        weights, architecture, mesh=mesh,
        config=ArtgorExactConfig(prefix_bm=2, head_bm=2, head_bk=2,
                                 head_bn=2, inference_chunk=2, parent_chunk=4),
        interpret=True,
    )
    states_d = jax.device_put(states, NamedSharding(mesh, P("core", None)))
    prepared_d = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P())), prepared,
    )
    actual = engine(states_d, prepared_d)
    expected = stream1_layernorm_reference_inference(states, weights, architecture)
    np.testing.assert_array_equal(actual, expected)
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run: `python -m pytest tests/test_artgor_exact_inference.py -q`

Expected: FAIL with `ModuleNotFoundError: tpu_beam_search.artgor_exact_inference`.

- [ ] **Step 3: Implement the stable engine and explicit fallback**

Use these public types and keep the calls separate:

```python
@dataclass(frozen=True)
class ArtgorExactConfig:
    prefix_bm: int = 4096
    head_bm: int = 256
    head_bk: int = 1024
    head_bn: int = 128
    dense_rounding: str = "late"
    inference_chunk: int = 32768
    parent_chunk: int = 131072

    def validate(self) -> None:
        if self.parent_chunk % self.inference_chunk:
            raise ValueError("parent_chunk must divide into whole inference chunks")
        if self.dense_rounding != "late":
            raise ValueError("the published exact engine requires late rounding")


@dataclass(frozen=True)
class EngineDecision:
    requested: str
    selected: str
    reason: str


@dataclass(frozen=True)
class ArtgorExactInference:
    prefix: Callable
    head: Callable

    def __call__(self, states, weights):
        hidden = self.prefix(states, weights)
        return self.head(hidden, weights)
```

Build the two local calls explicitly:

```python
prefix_local = lambda states, weights: stream1_layernorm_exact_prefix(
    states, weights, architecture,
    bm=config.prefix_bm, interpret=interpret,
)
head_local = lambda hidden, weights: pallas_layernorm_dense(
    hidden, weights.output.weight, weights.output.bias,
    bm=config.head_bm, bk=config.head_bk, bn=config.head_bn,
    dense_rounding=config.dense_rounding, interpret=interpret,
)
```

Compile each through its own `make_sharded_inference`; do not return a jitted composition.

- [ ] **Step 4: Run exact-inference tests and the existing exact suite**

Run: `python -m pytest tests/test_artgor_exact_inference.py tests/test_stream1_layernorm_exact.py tests/test_exact_inference_frontier.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the stable API**

```bash
git add src/tpu_beam_search/artgor_exact_inference.py src/tpu_beam_search/__init__.py tests/test_artgor_exact_inference.py
git commit -m "Expose exact Artgor TPU inference engine"
```

---

### Task 3: Implement the staged exact beam-depth executor

**Files:**
- Create: `src/tpu_beam_search/artgor_staged_beam.py`
- Create: `tests/test_artgor_staged_beam.py`
- Modify: `src/tpu_beam_search/__init__.py`

**Interfaces:**
- Consumes: the frozen Artgor solver contract and `ArtgorExactInference` from Task 2.
- Produces: `StagedDepthConfig`, `StagedDepthExecutables`, `build_staged_depth_executables`, `run_staged_depth`, and `beam_solve_v_only_spmd_packed_exact` with the original solver arguments plus `exact_config`.

- [ ] **Step 1: Write failing geometry and ordering tests**

```python
import numpy as np

from tpu_beam_search.artgor_staged_beam import (
    StagedDepthConfig,
    inference_chunk_starts,
    parent_window_starts,
    concatenate_q_chunks,
)


def test_default_depth_geometry_has_64_inference_chunks_and_16_search_windows():
    config = StagedDepthConfig(
        world_size=8, b_local=2_097_152, inference_chunk=32_768,
        parent_chunk=131_072, n_gen=30, state_size=150,
    )
    config.validate()
    assert len(inference_chunk_starts(config)) == 64
    assert inference_chunk_starts(config)[:4] == (0, 32768, 65536, 98304)
    assert inference_chunk_starts(config)[-1] == 2_064_384
    assert len(parent_window_starts(config)) == 16
    assert parent_window_starts(config)[-1] == 1_966_080


def test_q_parts_preserve_parent_then_move_order():
    parts = [np.full((1, 2, 3), value, np.float32) for value in range(4)]
    actual = concatenate_q_chunks(parts)
    assert actual.shape == (1, 8, 3)
    np.testing.assert_array_equal(actual[0, :, 0], [0, 0, 1, 1, 2, 2, 3, 3])
```

- [ ] **Step 2: Run the geometry tests and verify import failure**

Run: `python -m pytest tests/test_artgor_staged_beam.py -q`

Expected: FAIL with `ModuleNotFoundError: tpu_beam_search.artgor_staged_beam`.

- [ ] **Step 3: Implement validated static geometry**

```python
@dataclass(frozen=True)
class StagedDepthConfig:
    world_size: int
    b_local: int
    inference_chunk: int
    parent_chunk: int
    n_gen: int
    state_size: int

    def validate(self, *, require_published_geometry: bool = True) -> None:
        if self.b_local % self.parent_chunk:
            raise ValueError("parent_chunk must divide B_local")
        if self.parent_chunk % self.inference_chunk:
            raise ValueError("inference_chunk must divide parent_chunk")
        if require_published_geometry and (
            self.world_size != 8 or self.n_gen != 30 or self.state_size != 150
        ):
            raise ValueError("published Artgor exact geometry is 8 cores, 30 moves, 150 bytes")
```

Implement `inference_chunk_starts(config)` and `parent_window_starts(config)` as immutable tuples over `range(0, B_local, chunk_size)`. Implement `concatenate_q_chunks(parts)` as `jnp.concatenate(tuple(parts), axis=1)` for tensors shaped `[world_size, inference_chunk, MOVE_COUNT]`.

- [ ] **Step 4: Write the failing one-depth exactness test with an injected oracle engine**

Add a tiny deterministic fixture that constructs legal permutation states, the same move table, history carry, and Q values for both original and staged functions. Use `world_size=1`, `b_local=8`, `parent_chunk=4`, `inference_chunk=2`, `n_gen=2`, call `validate(require_published_geometry=False)`, and inject a callable oracle returning fixed BF16 Q. Compare every returned tensor:

```python
def test_staged_depth_matches_original_depth_tensor_for_tensor(tiny_depth_fixture):
    original = tiny_depth_fixture.run_original()
    staged = tiny_depth_fixture.run_staged()
    assert original.keys() == staged.keys()
    for name in original:
        np.testing.assert_array_equal(staged[name], original[name], err_msg=name)
```

The comparison keys must include `states`, `scores`, `owners`, `selected_ids`, `in_move`, `min_v_log`, `found_step`, `found_pos_local`, `found_pos_rank`, `verify_state`, `history`, and `packed_backpointer`.

- [ ] **Step 5: Run the one-depth test and verify it fails**

Run: `python -m pytest tests/test_artgor_staged_beam.py::test_staged_depth_matches_original_depth_tensor_for_tensor -q`

Expected: FAIL because `build_staged_depth_executables` and `run_staged_depth` are not implemented.

- [ ] **Step 6: Implement full-depth Q assembly and one unchanged search dispatch**

Define the compiled boundary object:

```python
@dataclass(frozen=True)
class StagedDepthExecutables:
    prefix_from_beam: Callable
    head: Callable
    assemble_q: Callable
    search_depth: Callable
```

`prefix_from_beam` receives the full `[world_size, B_local, 150]` sharded state tensor and a static parent offset, slices exactly 32,768 local rows inside its own prefix dispatch, and returns `[world_size, 32768, 1024]`. `head` is the separate selected Pallas dispatch and returns `[world_size, 32768, 30]`.

For each depth, queue all 64 ordered `prefix_from_beam` calls and their dependent `head` calls. `assemble_q` concatenates the BF16 outputs on the per-core parent axis into `[world_size, B_local, 30]`; no hidden tensors are stored. `search_depth` uses the original 131,072-parent `lax.scan`, slicing the corresponding Q window instead of calling `model_apply`. It retains the original child generation, `(parent_local, move)` flat IDs, inverse mask, hash/owner routing, per-destination top-K carry, `all_to_all`, history/dedup, final top-K, solved/endgame checks, and packed-backpointer logic.

No stage may call `np.asarray`, `jax.device_get`, or wrap `prefix_from_beam` and `head` in one compiled function.

- [ ] **Step 7: Run staged-depth tests**

Run: `python -m pytest tests/test_artgor_staged_beam.py -q`

Expected: PASS, including tensor-for-tensor one-depth equality.

- [ ] **Step 8: Add and test the high-level solver entry point**

Expose the original signature and one extra keyword:

```python
def beam_solve_v_only_spmd_packed_exact(
    init_state_list, v_params, all_moves, V0, hash_vec, mesh,
    B_local, K_per_peer, *, exact_config=ArtgorExactConfig(), **original_options,
) -> dict[str, Any]:
    exact_config.validate()
    return _beam_solve_staged(
        init_state_list=init_state_list,
        v_params=v_params,
        all_moves=all_moves,
        V0=V0,
        hash_vec=hash_vec,
        mesh=mesh,
        B_local=B_local,
        K_per_peer=K_per_peer,
        exact_config=exact_config,
        original_options=original_options,
    )
```

The returned dictionary must retain `found`, `path_len`, `path_idx`, `found_step`, `wall_s`, `first_iter_s`, `last_completed_step`, `lower_s`, `compile_s`, and `min_v_trajectory_rank0`, and add a `timing_breakdown` map without renaming existing keys.

Run: `python -m pytest tests/test_artgor_staged_beam.py tests/test_artgor_exact_inference.py -q`

Expected: PASS.

- [ ] **Step 9: Commit the staged executor**

```bash
git add src/tpu_beam_search/artgor_staged_beam.py src/tpu_beam_search/__init__.py tests/test_artgor_staged_beam.py
git commit -m "Add exact staged Artgor beam depth"
```

---

### Task 4: Build the attributed six-cell public notebook deterministically

**Files:**
- Create: `notebooks/artgor_cube555_exact_tpu/build_notebook.py`
- Create: `notebooks/artgor_cube555_exact_tpu/cayleypy-cube555-tpu-beam-q-exact.ipynb`
- Create: `notebooks/artgor_cube555_exact_tpu/kernel-metadata.json`
- Create: `tests/test_artgor_exact_notebook.py`

**Interfaces:**
- Consumes: the frozen notebook manifest, the mounted Artgor artifacts, the mounted code dataset, and `beam_solve_v_only_spmd_packed_exact`.
- Produces: a deterministic six-cell `.ipynb` and Kaggle metadata for `trydotatwo/cayleypy-cube555-tpu-beam-q-exact`.

- [ ] **Step 1: Write the failing notebook contract test**

```python
import ast
import json
from pathlib import Path


def test_generated_notebook_preserves_source_flow_and_adds_exact_engine():
    root = Path(__file__).resolve().parents[1]
    folder = root / "notebooks" / "artgor_cube555_exact_tpu"
    notebook = json.loads((folder / "cayleypy-cube555-tpu-beam-q-exact.ipynb").read_text())
    assert len(notebook["cells"]) == 6
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        "markdown", "code", "code", "code", "code", "code",
    ]
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    for source in sources[1:]:
        ast.parse(source)
    joined = "\n".join(sources)
    assert 'INFERENCE_ENGINE = "exact_split"' in joined
    assert "beam_solve_v_only_spmd_packed_exact" in joined
    assert "QV_CONSISTENCY" in joined and "BLEND_CHECKPOINTS" in joined
    assert "scriptVersionId=344319112" in joined
    assert "inference speedup" in joined.lower()
    assert "whole-solver speedup" in joined.lower()
```

- [ ] **Step 2: Run the notebook test and verify missing-file failure**

Run: `python -m pytest tests/test_artgor_exact_notebook.py -q`

Expected: FAIL because the generated notebook does not exist.

- [ ] **Step 3: Implement the deterministic builder**

The builder must:

```python
SOURCE_VERSION = 344319112
SOURCE_NOTEBOOK_SHA256 = "c74613a9fa400b391aca49bb128a2f6d3b0465e8e7cb933abc9b126a317e0e0b"
PUBLIC_SLUG = "trydotatwo/cayleypy-cube555-tpu-beam-q-exact"
```

Verify the frozen source hash before editing. Keep six cells. Add attribution and scoped performance wording to cell 0; add source/runtime manifest output to cell 1; add `INFERENCE_ENGINE` and `INFERENCE_CHUNK=32768` to cell 2; mount and verify the public code-dataset manifest before imports in cell 3; select the exact solver only for supported Q-only settings in cell 4; retain the original results/submission cell 5.

Generate metadata with TPU enabled, GPU disabled, internet disabled, the Artgor artifact dataset and code dataset attached, and the cube555 competition attached. Keep `is_private: true` until Task 8 passes the publication gate.

- [ ] **Step 4: Build twice and test determinism**

Run: `python notebooks/artgor_cube555_exact_tpu/build_notebook.py`

Run again: `python notebooks/artgor_cube555_exact_tpu/build_notebook.py --check`

Expected: the second command exits 0 and reports no byte changes.

- [ ] **Step 5: Run notebook validation tests**

Run: `python -m pytest tests/test_artgor_exact_notebook.py tests/test_artgor_source_snapshot.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the notebook builder and private draft**

```bash
git add notebooks/artgor_cube555_exact_tpu tests/test_artgor_exact_notebook.py
git commit -m "Build exact accelerated Artgor TPU notebook"
```

---

### Task 5: Package a self-contained public Kaggle code dataset

**Files:**
- Create: `kaggle_artgor_exact_code/dataset-metadata.json`
- Create: `kaggle_artgor_exact_code/build_release.py`
- Create: `tests/test_artgor_exact_code_package.py`

**Interfaces:**
- Consumes: a clean Git checkout containing the package, staged beam module, frozen manifest, and full source commit.
- Produces: `/tmp` or caller-selected release directory with `tpu_beam_search/`, `release_manifest.json`, and only the allowlisted runtime files; Kaggle dataset slug `trydotatwo/tpu-beam-search-exact-artgor-code`.

- [ ] **Step 1: Write the failing release-manifest test**

```python
def test_code_release_is_allowlisted_and_commit_pinned(tmp_path):
    result = build_release(tmp_path, source_commit="0" * 40)
    manifest = json.loads((result / "release_manifest.json").read_text())
    assert manifest["source_commit"] == "0" * 40
    assert manifest["artgor_script_version"] == 344319112
    assert set(manifest["files"]) == {
        "tpu_beam_search/__init__.py",
        "tpu_beam_search/artgor_exact_inference.py",
        "tpu_beam_search/artgor_staged_beam.py",
        "tpu_beam_search/sharding.py",
        "tpu_beam_search/stream1_architecture.py",
        "tpu_beam_search/stream1_embedding_experimental.py",
        "tpu_beam_search/stream1_layernorm_exact.py",
        "tpu_beam_search/stream1_layernorm_pallas.py",
        "tpu_beam_search/stream1_layernorm_reference.py",
        "tpu_beam_search/stream1_pallas.py",
        "tpu_beam_search/tpu_layout.py",
    }
```

- [ ] **Step 2: Run the package test and verify import failure**

Run: `python -m pytest tests/test_artgor_exact_code_package.py -q`

Expected: FAIL because `kaggle_artgor_exact_code.build_release` does not exist.

- [ ] **Step 3: Implement allowlisted packaging and hashes**

`build_release(output, source_commit)` must reject a non-40-hex commit outside tests, copy only the named runtime files, calculate SHA-256 for every output, and write the exact source commit, Artgor source version, Python/JAX/libtpu target versions, and file hashes to `release_manifest.json`. It must reject a nonempty output directory to prevent stale files entering a dataset version.

Dataset metadata:

```json
{
  "title": "TPUBeamSearch exact Artgor inference code",
  "id": "trydotatwo/tpu-beam-search-exact-artgor-code",
  "licenses": [{"name": "MIT"}]
}
```

The MIT declaration applies to TPUBeamSearch code only; `PROVENANCE.md` continues to attribute the separately mounted Artgor source/artifacts.

- [ ] **Step 4: Build and verify a release directory**

In PowerShell, run:

```powershell
$tpuPackageCommit = git rev-parse HEAD
python kaggle_artgor_exact_code/build_release.py --source-commit $tpuPackageCommit --output .release/artgor_exact_code
```

Expected: release manifest lists only allowlisted files and every hash verifies.

- [ ] **Step 5: Run package tests**

Run: `python -m pytest tests/test_artgor_exact_code_package.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the dataset packager**

```bash
git add kaggle_artgor_exact_code tests/test_artgor_exact_code_package.py
git commit -m "Package exact Artgor TPU runtime"
```

---

### Task 6: Add the one-job private TPU validation gate

**Files:**
- Create: `benchmarks/artgor_exact_notebook_validation.py`
- Create: `tests/test_artgor_exact_notebook_validation.py`
- Create after the benchmark-source commit: `kaggle_artgor_exact_validation/run_validation.py`
- Create after the benchmark-source commit: `kaggle_artgor_exact_validation/kernel-metadata.json`
- Create after the benchmark-source commit: `tests/test_artgor_exact_validation_launcher.py`

**Interfaces:**
- Consumes: frozen Artgor model/beam sources, checkpoint and puzzle assets, competition test states, exact engine, staged beam executor, and source-pinned launcher.
- Produces: `artgor_exact_notebook_validation.json`, incremental `validation.log`, solve artifacts, and a terminal `decision` object with immutable gate results.

- [ ] **Step 1: Write failing decision-gate tests**

```python
from benchmarks.artgor_exact_notebook_validation import decide_publication


def test_publication_requires_every_exact_and_replay_gate():
    passing = {
        "runtime": {"active_tpu_devices": 8},
        "inference": {"legal_exact": True, "stress_exact": True},
        "one_depth": {"all_tensor_hashes_equal": True},
        "short_solve": {"frontiers_equal": True, "backpointers_equal": True},
        "real_solve": {"pid": 1034, "sym": 0, "inverted": False,
                       "found": True, "verify_ok": True},
    }
    assert decide_publication(passing)["publishable"] is True
    passing["one_depth"]["all_tensor_hashes_equal"] = False
    assert decide_publication(passing)["publishable"] is False
```

- [ ] **Step 2: Run the benchmark decision test and verify import failure**

Run: `python -m pytest tests/test_artgor_exact_notebook_validation.py -q`

Expected: FAIL because the benchmark does not exist.

- [ ] **Step 3: Implement incremental validation output**

The benchmark must checkpoint JSON after each phase and abort before timing promotion when exactness fails. Record:

```python
report = {
    "status": "running",
    "context": {
        "source_commit": source_commit,
        "artgor_script_version": 344319112,
        "runtime": runtime_inventory(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_source_sha256": sha256_file(jax_model_path),
        "beam_source_sha256": sha256_file(jax_beam_path),
        "puzzle_sha256": sha256_file(puzzle_path),
    },
    "inference": {}, "one_depth": {}, "short_solve": {},
    "real_solve": {}, "timings": {}, "decision": {},
}
```

Inference uses 32,768 states per device on eight devices and the same legal/stress seeds and hashes as the exact-frontier experiment. One-depth validation records each named tensor's shape, dtype, reference hash, candidate hash, mismatch count, and first mismatch witness. The short A/B uses a bounded beam/depth that completes in one session. The real run uses `pid=1034`, frame 0, no inversion, `B_GLOBAL=16,777,216`, `B_LOCAL=2,097,152`, `PARENT_CHUNK=131072`, and replays the complete path against the original competition state.

- [ ] **Step 4: Implement paired timing without overstating results**

Record compile, first call, steady prefix/head, selection, communication/dedup, backpointer copy/write, complete depth, and solve wall time. `decision["inference_speedup"]` compares exact full Q forwards; `decision["beam_depth_speedup"]` exists only after a same-runtime original/staged pair; `decision["solver_speedup"]` exists only after comparable completed solves. Omit unavailable speedups rather than filling them from old runs.

- [ ] **Step 5: Commit the benchmark source before creating its launcher**

```bash
git add benchmarks/artgor_exact_notebook_validation.py tests/test_artgor_exact_notebook_validation.py
git commit -m "Add exact Artgor notebook TPU validation"
```

Resolve the immutable benchmark source with `git rev-parse HEAD` and retain the returned 40-hex value for the next step.

- [ ] **Step 6: Add the source-pinned private launcher**

Follow existing launchers. Set `COMMIT_SHA` to the exact 40-hex output from Step 5; the launcher test must assert the same literal value. The remaining constants are:

```python
REPOSITORY = "https://github.com/TryDotAtwo/TPUBeamSearch.git"
OUTPUT = Path("/kaggle/working/artgor_exact_notebook_validation")
```

The launcher installs `jax[tpu]==0.10.2`, `jaxlib==0.10.2`, and `libtpu==0.0.42.1`; clones and detaches the pinned commit; verifies `git rev-parse HEAD`; streams stdout to both Kaggle and `validation.log`; and invokes `python -m benchmarks.artgor_exact_notebook_validation`.

Metadata must be private, TPU-only, internet-enabled for the pinned clone, attach `artgor/cube555-tpu-artifacts`, and attach the `cayley-py-555-cube` competition.

Create `tests/test_artgor_exact_validation_launcher.py` to JSON-parse metadata, AST-parse the launcher, assert the literal `COMMIT_SHA` from Step 5, require a full 40-hex SHA, and verify the repository URL, TPU/JAX versions, module name, output directory, streaming log, detached checkout, and both Kaggle data sources.

- [ ] **Step 7: Run validation/launcher tests**

Run: `python -m pytest tests/test_artgor_exact_notebook_validation.py tests/test_artgor_exact_validation_launcher.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the pinned launcher**

The launcher points to the benchmark-source commit from Step 5. Commit the launcher and its literal-SHA test separately:

```bash
git add kaggle_artgor_exact_validation/run_validation.py tests/test_artgor_exact_validation_launcher.py
git commit -m "Pin exact Artgor validation source"
```

---

### Task 7: Verify locally and publish the validation source

**Files:**
- Modify only files from Tasks 1-6 if a local test exposes a defect.

**Interfaces:**
- Consumes: all implementation tasks and their commits.
- Produces: one clean, public Git SHA that the private Kaggle launcher can clone.

- [ ] **Step 1: Rebuild generated artifacts and verify no drift**

Run: `python notebooks/artgor_cube555_exact_tpu/build_notebook.py --check`

In PowerShell, run:

```powershell
$tpuNotebookCommit = git rev-parse HEAD
$tpuReleaseDirectory = Join-Path $env:TEMP ("tpu-artgor-code-" + $tpuNotebookCommit.Substring(0, 12))
New-Item -ItemType Directory -Path $tpuReleaseDirectory
python kaggle_artgor_exact_code/build_release.py --source-commit $tpuNotebookCommit --output $tpuReleaseDirectory
```

Expected: notebook check reports no changes; release manifest hashes all verify.

- [ ] **Step 2: Run focused integration tests**

Run: `python -m pytest tests/test_artgor_source_snapshot.py tests/test_artgor_exact_inference.py tests/test_artgor_staged_beam.py tests/test_artgor_exact_notebook.py tests/test_artgor_exact_code_package.py tests/test_artgor_exact_notebook_validation.py tests/test_artgor_exact_validation_launcher.py -q`

Expected: PASS.

- [ ] **Step 3: Run the complete project suite**

Run: `python -m pytest -q`

Expected: every test passes; record the exact count and elapsed time.

- [ ] **Step 4: Audit the scoped diff and push main**

Run: `git status --short`, `git diff --check`, and `git log --oneline origin/main..HEAD` separately.

Verify that unrelated untracked logs remain unstaged. Push the ordinary scoped commits directly to `origin/main`; do not force-push or rewrite history.

---

### Task 8: Run the private eight-TPU gate and correct only reproducible failures

**Files:**
- Create after terminal run: `test_results/kaggle_artgor_exact_notebook_validation_v1/` or the actual versioned result directory.
- Modify implementation/test files only when a downloaded failure is reproducible and covered by a failing test.

**Interfaces:**
- Consumes: pinned public Git SHA and one free Kaggle TPU session.
- Produces: terminal private validation evidence that either passes every publication gate or precisely rejects the notebook.

- [ ] **Step 1: Confirm no active TPU job and push the private kernel**

Run: `kaggle kernels list --mine --page-size 100` and inspect recent TPU kernels.

Run: `kaggle kernels push -p kaggle_artgor_exact_validation`

Expected: slug `trydotatwo/tpu-artgor-exact-notebook-validation` enters QUEUED or RUNNING with exactly one TPU session.

- [ ] **Step 2: Monitor without duplicate restarts**

Run: `kaggle kernels status trydotatwo/tpu-artgor-exact-notebook-validation` at bounded intervals. Do not restart QUEUED/RUNNING. If this work extends beyond the current interactive run, create one heartbeat automation that checks this slug and deletes itself after terminal publication or rejection.

- [ ] **Step 3: Download terminal output and verify the machine gate**

Run: `kaggle kernels output trydotatwo/tpu-artgor-exact-notebook-validation -p test_results/kaggle_artgor_exact_notebook_validation_v1`

Require `status == "complete"` and `decision.publishable == true`. Independently recompute inference mismatch counts, one-depth hashes, short-frontier hashes, and host replay from the downloaded JSON/path rather than trusting the summary boolean alone.

- [ ] **Step 4: Handle a terminal error through TDD**

If the kernel errors or any gate rejects, preserve JSON/logs, identify the first failing phase and exact exception/mismatch witness, add one focused failing local test reproducing that cause, run it to confirm failure, implement the smallest fix, run focused plus full suites, scoped commit/push, update the pinned launcher SHA, and restart only this validation kernel. Do not weaken exactness thresholds or change search semantics after seeing results.

- [ ] **Step 5: Write and commit the private validation report**

The report must state runtime/device inventory, source/checkpoint/source/input hashes, exactness for both corpora, one-depth tensor gate, short A/B, real `pid=1034` path/replay, inference speedup, any measured beam-depth/solver speedup, and timing limitations. Include the safe JSON and useful log; exclude credentials and unrelated files.

---

### Task 9: Publish the code dataset and public Kaggle notebook

**Files:**
- Modify: `notebooks/artgor_cube555_exact_tpu/kernel-metadata.json`
- Modify: `docs/research/tpu_experiment_ledger.md`
- Create: `test_results/kaggle_artgor_exact_notebook_public_v1/report.md`
- Create: `test_results/kaggle_artgor_exact_notebook_public_v1/artifact_manifest.json`

**Interfaces:**
- Consumes: a passing private validation result and the exact Git SHA it tested.
- Produces: public Kaggle code dataset, public copy-and-run notebook, GitHub report/ledger entry, and verifiable slugs/versions.

- [ ] **Step 1: Build and publish the versioned code dataset**

Resolve the exact Git SHA recorded by the passing private validation JSON, then build a new release directory:

```powershell
$tpuValidatedCommit = (Get-Content -Raw 'test_results\kaggle_artgor_exact_notebook_validation_v1\artgor_exact_notebook_validation.json' | ConvertFrom-Json).context.source_commit
$tpuPublicRelease = Join-Path $env:TEMP ("tpu-artgor-public-" + $tpuValidatedCommit.Substring(0, 12))
New-Item -ItemType Directory -Path $tpuPublicRelease
python kaggle_artgor_exact_code/build_release.py --source-commit $tpuValidatedCommit --output $tpuPublicRelease
```

Run `kaggle datasets create -p $tpuPublicRelease` for the first version or `kaggle datasets version -p $tpuPublicRelease -m ("Exact Artgor TPU notebook runtime " + $tpuValidatedCommit.Substring(0, 12))` for an existing dataset. Verify the returned dataset slug and version and download/list it to confirm `release_manifest.json` and every file hash.

- [ ] **Step 2: Pin the dataset version and make notebook metadata public**

Update the notebook mount/manifest expectation to the published code-dataset version, set `is_private` to `false`, keep TPU enabled and internet disabled, rebuild the notebook, and rerun notebook JSON/AST/determinism tests.

- [ ] **Step 3: Push the public notebook and run its published version**

Run: `kaggle kernels push -p notebooks/artgor_cube555_exact_tpu`

Monitor the public slug to COMPLETE. Download its outputs and require the same manifest SHA, exact preflight, and replay-valid default validation artifact. A successful private script alone does not substitute for running the actual published notebook package.

- [ ] **Step 4: Publish the final report and ledger update**

Record Git SHA, public dataset slug/version, public notebook slug/version/status, device/runtime inventory, inference and end-to-end measurements with their scopes, exactness hashes, solution path length/replay, and links. State explicitly whether the whole solver achieved a measured speedup; retain the `~1.6x` wording only for the exact inference scope if no comparable solver measurement exists.

- [ ] **Step 5: Run final verification and push scoped release commits**

Run generated-notebook checks, focused release tests, and the full project suite. Audit `git status --short`, stage only notebook metadata/generated notebook/report/ledger/result JSON/allowlisted log, commit, and ordinary-push `main`.

The task is complete only when GitHub and Kaggle both expose the tested artifacts and the public notebook's actual terminal output passes the publication gate.
