# Arithmetic Follow-up Bundle Implementation Plan

> **For agentic workers:** Use TDD for each component and independently review the integrated experiment before launch.

**Goal:** Test predicate layout, HLO-informed LN arithmetic, and full-forward late Dense independently in one queued TPU job.

**Architecture:** Add an experimental kernel module and a new benchmark, leaving the v1 benchmark and production defaults unchanged. Share already established reference/quality helpers. Timing helpers accept already compiled executables, separating diagnostic profiling from promotion.

**Tech Stack:** Python, JAX/Pallas; local CPU interpreter; Kaggle JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

**Spec:** `docs/research/2026-08-31-arithmetic-followup-research.md`; user approved the proposed bundle on 2026-08-31.

## Global constraints

- BN and production LN defaults remain unchanged.
- Model: embedding 150x24, state150, hidden1024, ten residual blocks, Q30, minimize.
- Accept only finite, elementwise-exact original monolithic Q on both anchor corpora; 16K qualification and 32K confirmation remain separate.
- Preserve runtime parameters and original corpus hashes. Global top-K is a proxy, not distributed beam validation.
- CPU passes establish interpreter behavior, not TPU lowering or speed.
- Public GitHub source SHA first, then private Kaggle launch; one TPU session at a time.

## Task 1: Experimental kernels

Files: `src/tpu_beam_search/stream1_layernorm_experimental.py`, `tests/test_layernorm_experimental.py`.

- [x] Write independent-expression tests for masked BF16 and mixed FP32/BF16 LN, widths1024/130, and unsafe partial-mask rejection.
- [x] Observe failing tests before implementing `experimental_layer_norm` and `minimal_predicate_select`.
- [x] Implement mask-site isolation, FP32 select operands, direct 2-D predicates and mixed arithmetic with explicit rounding points.
- [x] Run `python -m pytest tests/test_layernorm_experimental.py -q`; inspect padding tails and unchanged defaults.

## Task 2: Timing without graph changes

Files: `benchmarks/diagnostic_timing.py`, `tests/test_diagnostic_timing.py`.

- [x] Write failing tests using real CPU compiled executables.
- [x] Implement `paired_interleaved_measure`, `queued_measure`, `diagnostic_profile`.
- [x] Synchronize every output; retain all queued outputs. Label queued calls as not a real chunk scan.
- [x] Run `python -m pytest tests/test_diagnostic_timing.py -q`.

## Task 3: Bundle orchestration and acceptance

Files: `benchmarks/stream1_layernorm_followup.py`, `tests/test_layernorm_followup.py`.

- [x] Write failing tests for complete-forward runtime controls, late Dense/JAX LN, rejected-candidate profiles, gate and partial-failure persistence.
- [x] Implement minimal synthetic probes, same-input LN/block comparisons and matching same-suffix controls, then full16K cases on both original corpora.
- [x] Save StableHLO before compilation and compiled HLO afterward, including per-case errors. Measure interleaved warmed calls and queued calls without adding a JIT/scan boundary.
- [x] Promote at most two non-control candidates exact on both corpora to32K. Profile compiled cases independently of acceptance; keep rejected speedups null.
- [x] Run targeted tests, then `python -m pytest -q` and independent review. Final local result:252 passed; review fixes rechecked.

## Task 4: Publish and launch

Files: `kaggle_layernorm_followup/`, `test_results/kaggle_layernorm_followup_v1/launch.md`, experiment ledger.

- [ ] Validate launcher AST/metadata; publish scoped source commit main.
- [ ] Pin the full published SHA in the launcher; publish that launcher commit.
- [ ] Verify no active TPU session, push exactly one private kernel, and read actual status/version.
- [ ] Record requested/effective runtime distinction, launch provenance, and monitoring instructions. Keep queued/running jobs intact.
