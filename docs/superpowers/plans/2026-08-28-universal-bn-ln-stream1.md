# Universal BN/LN Stream1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add correctness-gated LayerNorm ResMLP inference to the universal Stream1 engine and select its TPU input encoding and Pallas tiling through reproducible A/B benchmarks.

**Architecture:** Preserve the current folded-BN execution while extending its static architecture contract with normalization and input-encoding plans. Reproduce Andrey's JAX model first, then add Pallas LayerNorm and encoding candidates, measuring each at identical shapes before selecting a production plan.

**Tech Stack:** Python, JAX, Pallas/Mosaic TPU, NumPy, pytest, Kaggle TPU VM v3-8.

**Spec:** `docs/superpowers/specs/2026-08-28-universal-bn-ln-stream1-design.md`

## Global Constraints

- Existing folded-BatchNorm inference and its selected tiling must remain unchanged unless a correctness-gated regression benchmark proves equivalence.
- LayerNorm reductions and normalization arithmetic use FP32; storage and dense layer boundaries use BF16.
- `EMBEDDING_GATHER`, `VIRTUAL_ONE_HOT_MXU`, and `FUSED_VIRTUAL_ONE_HOT` remain candidates until complete-network TPU measurements select one.
- Never materialize the full virtual one-hot tensor in HBM.
- Compile time is excluded and every timed boundary calls `block_until_ready()`.
- All architecture dimensions come from configuration/checkpoint validation; use names such as `MOVE_COUNT`, never unexplained numeric literals.

---

### Task 1: Lock down the original LayerNorm oracle and baseline

**Files:**
- Modify: `benchmarks/artgor_cube555_mlp.py`
- Create: `tests/test_artgor_reference.py`
- Create: `test_results/artgor_cube555_mlp_baseline/README.md`

**Interfaces:**
- Consumes: the inspected public `jax_model.py` semantics and checkpoint tensor names.
- Produces: `artgor_reference_apply(states, weights, architecture) -> jax.Array` and JSON records keyed by `device_count`, `local_batch`, `latency_ms`, and `states_per_second`.

- [ ] **Step 1: Write failing oracle tests** for the exact embedding, first LayerNorm, ten residual blocks, and Q-head using a tiny shape-compatible deterministic fixture.
- [ ] **Step 2: Run `pytest tests/test_artgor_reference.py -v`** and verify failure because the reusable oracle interface is absent.
- [ ] **Step 3: Extract the exact original apply semantics** into benchmark-importable functions without algebraic rewrites; retain `epsilon`, operation ordering, and BF16 casts.
- [ ] **Step 4: Run the oracle tests** and require exact shape, finite values, and deterministic repeated output.
- [ ] **Step 5: Run/download the private Kaggle baseline** at local batches 16384 and 32768 plus the real 128-chunk scan shape; preserve JSON and the full private log locally.
- [ ] **Step 6: Commit only source, tests, and publishable summary artifacts** with `git add` scoped to the named files and `git commit -m "test: establish Artgor LayerNorm MLP baseline"`.

### Task 2: Extend the architecture contract without regressing BN

**Files:**
- Modify: `src/tpu_beam_search/stream1_architecture.py`
- Modify: `src/tpu_beam_search/stream1_inference.py`
- Create: `tests/test_stream1_architecture.py`
- Modify: `tests/test_stream1_inference.py`

**Interfaces:**
- Produces: `NormalizationKind`, `InputEncodingKind`, a shape-derived LayerNorm ResMLP architecture, `LayerNormWeights(scale, bias)`, and weights capable of representing embedding plus normalized residual layers.
- Preserves: `Stream1Architecture`, `Stream1Weights`, and current BN checkpoint conversion behavior.

- [ ] **Step 1: Add failing tests** proving old BN construction is source-compatible and LayerNorm checkpoints derive `STATE_LEN`, `NUM_CLASSES`, `EMBED_DIM`, `HIDDEN`, `RESIDUAL_COUNT`, and `MOVE_COUNT` with validation errors for inconsistent shapes.
- [ ] **Step 2: Run the two targeted test files** and verify only the new contract tests fail.
- [ ] **Step 3: Add static enums and focused weight records** while retaining compatibility constructors/aliases for the existing BN callers.
- [ ] **Step 4: Implement exact checkpoint shape validation** for embedding, input projection, both dense layers per residual block, all LayerNorm affine vectors, and Q-head.
- [ ] **Step 5: Run `pytest tests/test_stream1_architecture.py tests/test_stream1_inference.py -v`** and require all old and new tests to pass.
- [ ] **Step 6: Commit** with `git commit -m "feat: add universal BN and LayerNorm architecture contract"` after scoped staging.

### Task 3: Implement the LayerNorm JAX reference path

**Files:**
- Create: `src/tpu_beam_search/stream1_layernorm_reference.py`
- Modify: `src/tpu_beam_search/stream1_inference.py`
- Create: `tests/test_stream1_layernorm_reference.py`

**Interfaces:**
- Produces: `layer_norm_fp32(values, scale, bias, epsilon) -> jax.Array` and `stream1_layernorm_reference_inference(states, weights, architecture, input_encoding) -> jax.Array`.
- Consumes: Task 2 architecture and weights; Task 1 oracle.

- [ ] **Step 1: Write failing tests** comparing LayerNorm statistics and complete output against the Task 1 oracle on deterministic random and edge categorical states.
- [ ] **Step 2: Run the new test file** and verify the missing reference functions cause failure.
- [ ] **Step 3: Implement embedding-gather reference inference** with FP32 mean/variance/rsqrt and exact residual operation ordering.
- [ ] **Step 4: Implement mathematical reference forms** for virtual one-hot and fused virtual-one-hot without allocating a dense `[B, STATE_LEN, NUM_CLASSES]` array.
- [ ] **Step 5: Run correctness tests** and record max/mean absolute error and argmax agreement for all encoding modes.
- [ ] **Step 6: Run the complete existing suite** with `pytest -q` and require no BN regression.
- [ ] **Step 7: Commit** with `git commit -m "feat: add LayerNorm ResMLP reference inference"`.

### Task 4: Build a fair input-prefix A/B benchmark

**Files:**
- Create: `benchmarks/stream1_layernorm_input_ab.py`
- Create: `tests/test_stream1_layernorm_input_ab.py`
- Create: `kaggle_layernorm_input_ab/kernel-metadata.json`
- Create: `kaggle_layernorm_input_ab/run.py`

**Interfaces:**
- Produces: per-candidate correctness and timing JSON for `EMBEDDING_GATHER`, `VIRTUAL_ONE_HOT_MXU`, and `FUSED_VIRTUAL_ONE_HOT`.

- [ ] **Step 1: Write failing contract tests** requiring identical seeded inputs/weights, explicit warmup/timed iterations, synchronization, local/global batches, and correctness fields in every candidate record.
- [ ] **Step 2: Run the contract test** and verify failure while the benchmark is absent.
- [ ] **Step 3: Implement the benchmark harness** with compilation outside timing and both prefix-only and complete-reference timing sections.
- [ ] **Step 4: Add the private Kaggle runner** that clones an exact source commit and writes one result JSON.
- [ ] **Step 5: Run local contract tests** and validate emitted JSON schema without requiring a TPU.
- [ ] **Step 6: Commit** with `git commit -m "bench: add LayerNorm input encoding comparison"`.

### Task 5: Implement and tune Pallas LayerNorm

**Files:**
- Create: `src/tpu_beam_search/stream1_layernorm_pallas.py`
- Create: `tests/test_stream1_layernorm_pallas.py`
- Create: `benchmarks/stream1_layernorm_tiling.py`

**Interfaces:**
- Produces: `pallas_layer_norm(values, scale, bias, *, bm, bn, epsilon, interpret=False) -> jax.Array` and fused dense/LN/ReLU variants used by Task 6.

- [ ] **Step 1: Write failing interpret-mode tests** for constant rows, large-offset rows, non-multiple logical widths with aligned storage, and affine scale/bias.
- [ ] **Step 2: Run the tests** and verify failure because the kernel is absent.
- [ ] **Step 3: Implement FP32 tiled row reductions** with masked aligned tails and BF16 output.
- [ ] **Step 4: Add dense→LN and dense→LN→ReLU fusion candidates** without changing the standalone correctness oracle.
- [ ] **Step 5: Run interpret tests and the BN suite**; establish an explicit numerical gate from observed BF16 error rather than weakening assertions after benchmarking.
- [ ] **Step 6: Add a TPU tiling sweep** over aligned `bm`/`bn` candidates for width 1024 and production batch shapes.
- [ ] **Step 7: Commit** with `git commit -m "feat: add Pallas LayerNorm kernels"`.

### Task 6: Implement three Pallas input-prefix candidates

**Files:**
- Modify: `src/tpu_beam_search/stream1_layernorm_pallas.py`
- Modify: `benchmarks/stream1_layernorm_input_ab.py`
- Modify: `tests/test_stream1_layernorm_pallas.py`

**Interfaces:**
- Produces: `pallas_layernorm_input_prefix(..., input_encoding: InputEncodingKind, ...) -> jax.Array` returning normalized/relu hidden `[B,HIDDEN]`.

- [ ] **Step 1: Add failing candidate-equivalence tests** against Task 3 for gather, virtual one-hot MXU, and fused virtual one-hot.
- [ ] **Step 2: Run tests** and verify all missing Pallas modes fail.
- [ ] **Step 3: Implement gather baseline** with aligned embedding/flatten/dense buffers.
- [ ] **Step 4: Implement virtual one-hot MXU** without an HBM one-hot allocation.
- [ ] **Step 5: Implement the fused candidate** that avoids writing `[B, STATE_LEN*EMBED_DIM]` when its block mapping is valid.
- [ ] **Step 6: Run correctness gates and local benchmark contract tests** for every candidate.
- [ ] **Step 7: Commit** with `git commit -m "feat: add Pallas LayerNorm input candidates"`.

### Task 7: Complete and optimize LayerNorm residual inference

**Files:**
- Modify: `src/tpu_beam_search/stream1_layernorm_pallas.py`
- Modify: `src/tpu_beam_search/stream1_inference.py`
- Create: `benchmarks/stream1_layernorm_full_mlp.py`
- Modify: `tests/test_stream1_layernorm_pallas.py`

**Interfaces:**
- Produces: the LayerNorm branch of `make_jitted_stream1_inference(...)`, executing input prefix, configurable residual stack, and Q-head.

- [ ] **Step 1: Write failing tests** for one residual block, ten blocks, Q-head output, static normalization specialization, and invalid mode combinations.
- [ ] **Step 2: Run tests** and verify the complete LayerNorm Pallas path is missing.
- [ ] **Step 3: Implement one block** as dense→LN→ReLU, dense→LN, skip→ReLU using Task 5 kernels.
- [ ] **Step 4: Extend to arbitrary configured residual count and the aligned Q-head** while preserving the BN dispatch unchanged.
- [ ] **Step 5: Add separate, per-block, and paired fusion variants** only where they share the exact reference semantics.
- [ ] **Step 6: Run LayerNorm correctness tests and the full existing suite**.
- [ ] **Step 7: Commit** with `git commit -m "feat: complete Pallas LayerNorm ResMLP inference"`.

### Task 8: Run decisive 1/8-TPU comparisons and select production defaults

**Files:**
- Create: `kaggle_layernorm_full_comparison/kernel-metadata.json`
- Create: `kaggle_layernorm_full_comparison/run.py`
- Create: `test_results/kaggle_layernorm_full_comparison_2026-08-28.md`
- Modify: `src/tpu_beam_search/stream1_inference.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: original JAX oracle and every correctness-valid Pallas candidate.
- Produces: measured production encoding/tiling defaults and a reproducible comparison report.

- [ ] **Step 1: Add a private Kaggle runner** for original JAX, our JAX reference, and every valid Pallas candidate at identical 1-device and 8-device shapes.
- [ ] **Step 2: Validate runner provenance and JSON schema locally**, including exact Git commit, TPU topology, JAX version, dtype, warmups, iterations, and correctness metrics.
- [ ] **Step 3: Push the exact source commit and launch the private kernel once**; do not restart while queued or running.
- [ ] **Step 4: Download terminal JSON and full log**, preserving private raw logs locally and publishing only safe result artifacts.
- [ ] **Step 5: Compute latency, states/s, speedup, and parallel efficiency** and reject any result that failed its correctness gate.
- [ ] **Step 6: Select the production input encoding and tiling** from complete-network throughput; when statistically indistinguishable, select the simpler valid implementation.
- [ ] **Step 7: Encode measured defaults and document the evidence**, then run `pytest -q` once more.
- [ ] **Step 8: Commit and scoped-push** source, tests, JSON-safe results, and report with `git commit -m "perf: select LayerNorm ResMLP TPU inference"`.
