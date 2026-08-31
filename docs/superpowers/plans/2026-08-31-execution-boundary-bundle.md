# Execution boundary bundle implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Follow TDD and review each deliverable.

**Goal:** Explain and reduce the measured JAX/Pallas gap with one reproducible TPU job.

**Architecture:** New opt-in modules only. Reuse existing numerical gates, runtime payloads, compiled interleaved timing and profiling. Separate Dense boundaries/tiles, LN predicates, and embedding; combine only after individual measurements.

**Tech Stack:** Python, JAX/jaxlib0.10.2, libtpu0.0.42.1, Pallas; local CPU interpretation is not TPU evidence.

**Spec:** `docs/research/2026-08-31-jax-pallas-execution-attribution.md`; user approved the proposed three-part bundle with “Делай дальше”.

## Global Constraints

- One TPU session at a time; do not restart QUEUED/RUNNING jobs.
- User explicitly requested no subagents during implementation. Remaining work
  and review run inline; independent agent review is not claimed.
- GitHub source first, pinned SHA in private Kaggle launcher; scoped direct push main authorized.
- Preserve BN/defaults and unrelated untracked artifacts. Work in the existing approved main checkout.
- Checkpoint Artgor cube555: state150, categories150, embedding24, hidden1024, ten residual blocks, Q30 minimized.
- Runtime FP32 original parameters and explicitly labeled BF16 control; unchanged legal/stress32768 hashes.
- Finite elementwise exact monolithic Q on both16K corpora; actual32K confirmation separate. Rejected cases may be profiled but never promoted as speedups.
- All padding neutral; no LN unmasked mode unless logical width equals storage width.
- Device timings exclude compile/transfer, use interleaving and retained queued outputs; queued calls are not real128chunk scan.

### Task 1: Flat embedding operators

**Files:** Create `src/tpu_beam_search/stream1_embedding_experimental.py`, `tests/test_embedding_experimental.py`.

**Interface:** `flat_embedding(states, embedding, *, implementation, bm=128, interpret=False)` returns BF16 `[B,state_len*embed_dim]`. Implementations `jax_flat`, `jax_tiled`, `pallas_banked`; reference remains ordinary JAX gather/reshape.

- [ ] Write tests first for position-major lookup, classes128..149, odd row/state dimensions, runtime table mutation and invalid dtype/shape/implementation.

```python
# Independent expected values: state0 picks row2 then row0, not interleaved dimensions.
states = jnp.array([[2, 0]], dtype=jnp.uint8)
table = jnp.array([[1., 2.], [3., 4.], [5., 6.]], dtype=jnp.float32)
np.testing.assert_array_equal(flat_embedding(states, table, implementation="jax_flat"), [[5.,6.,1.,2.]])
```

- [ ] Run tests and observe missing-behavior failure, then implement JAX flat indexing and row-tiled reference gather.
- [ ] Add the real Pallas flat-output candidate with bounded category banks, FP32 selection and BF16 output. Preserve exact element lookup, no embedding/Dense contraction. Keep implementation generic where practical, reject unsupported geometry explicitly; target150x24 must be covered.
- [ ] Test interpreter versus independent lookup including class149, position boundary128 and output tail3600; no signed-int8 conversion. Run focused tests and self-review. Root handles commits.

### Task 2: Boundary and numerical controls

**Files:** Create `benchmarks/execution_boundary_ops.py`, `tests/test_execution_boundary_ops.py`.

**Interfaces:** `dense_configs()`, `full_configs()`, `candidate_dense(x,w,b,config,interpret=False)`, `candidate_full(config,architecture,interpret=False)` returning a runtime `(states,weights)` callable; `mismatch_witnesses(reference,candidate,limit=16)` returns strict-JSON finite values/coordinates/dtypes.

- [ ] Write failing tests: JAX full control matches original graph, changed runtime head/table changes output, pre/post/both barrier modes survive lowering, unknown modes rejected, witnesses include shape and nonfinite safely.

```python
got = mismatch_witnesses(np.array([[1.,2.]],np.float32), np.array([[1.,3.]],np.float32))
assert got["mismatch_count"] == 1
assert got["examples"][0]["index"] == [0,1]
```

- [ ] Implement Dense-only BM128/256/512 fixed BK256BN512; BN1024 separate; BK1024 explicitly joint arithmetic/schedule. Use existing late Dense without editing its default.
- [ ] Full arms: JAX graph control; JAX Dense barriers pre/post/both; each Dense tile with JAX LN; JAX Dense plus mixed LN none/direct2D; each flat embedding with all-JAX trunk. No cross-product explosion.
- [ ] Add instrumented JAX LN/Dense-LN component outputs and same-expression instrumentation control. Preserve variance estimator, BF16 boundaries and epsilon; label observation-induced graph changes.
- [ ] Focused CPU tests, including interpreter arms and full graph control, then task review.

### Task 3: One reproducible experiment runner

**Files:** Create `benchmarks/stream1_execution_boundary.py`, `tests/test_execution_boundary_benchmark.py`.

**Consumes:** Task1/2 APIs; reuse `compile_case`, `measure_comparison_group`, `finalize_eligible_speedups`, `promotion_candidates`, runtime/checkpoint/puzzle helpers and `diagnostic_timing`.

- [ ] Write failing tiny-run test with real CPU models, one malformed candidate, two corpora, partial JSON and unchanged exactness gate.
- [ ] Measure matched Dense and Dense+LN arms on the same4096 hidden input; retain independently compiled Dense feeding the same LN reference. Save direct mismatch witnesses, not aggregate equality claims.
- [ ] Measure mixed LN none/direct2D and existing masked controls on identical Dense values. Compare instrumented JAX LN output to uninstrumented JAX; statistics diagnostics cannot certify the original graph when this control differs.
- [ ] Measure flat embeddings against runtime-FP32 reference and typed control at4096, then candidate full16K models with original/typed baselines. Preserve compile failures; no timings for rejected compiles.
- [ ] Compile before paired timing, profile legal16K candidates independently of Q gate, retain eight queued outputs; conditionally test at most two exact candidates at32K.
- [ ] Validate tiny-run strict JSON, gate, fallback comparability, shape/parameter sensitivity; full tests and inline whole-change review (user requested no subagents).

### Task 4: Publication and launch

**Files:** Create `kaggle_execution_boundary/run_execution_boundary.py`, `kaggle_execution_boundary/kernel-metadata.json`; update research ledger and launch record.

- [ ] Adapt the existing Git bootstrap, pinned dependencies, checkpoint location and output preservation. Test AST/metadata and CLI help locally.
- [ ] Scoped commit/push source, verify remote SHA, pin launcher and publish launcher. Check all owned kernels for a free slot before launch.
- [ ] Launch private `trydotatwo/tpu-execution-boundary-ab`, record returned version and live status. Reuse configured proxy.
- [ ] Create a thread heartbeat for terminal download/analysis/publication and deletion after final report. Existing completed monitors remain deleted.

## Execution record

Plan approved from the preceding discussion; no new design approval needed.
Shared interfaces: Task1 flat function is consumed only by Task2 full builder and Task3 isolated embedding suite. Task2 configs/builders are consumed only by Task3. Task4 only launches Task3. File ownership is disjoint.

Tasks1-3 implemented and reviewed inline. Fresh verification:281 tests passed
in102.81s. Known CPU/TPU gather-contract mismatch was reproduced by a failing
test and corrected; target compilation is still unverified. Tiny suites execute
promotion to a genuinely larger batch, preserve compile failures and fatal
partial reports, and reject output overwrite. Protocol frozen in
`docs/research/2026-08-31-execution-boundary-bundle.md`.

Task4 publication is in progress. Live Kaggle status/list requests currently
return HTTP403 via the configured proxy, outside the sandbox and in an isolated
direct read. No new kernel has been submitted; unknown TPU occupancy is not
treated as a free slot. No system proxy or VPN setting was changed.

Task4 completed on2026-08-31 after access recovered: version1 submitted once,
QUEUED confirmed17:14UTC, server source/private metadata checked, launch record
in `test_results/kaggle_execution_boundary_v1/`. The existing heartbeat was
updated to monitor the submitted version. Runtime results remain pending.
