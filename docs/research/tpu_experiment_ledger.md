# TPUBeamSearch experiment ledger

This file separates measured facts from hypotheses so benchmark conclusions do
not silently turn into architecture assumptions.

## Audit correction: 2026-08-31

Read [the source/code audit](2026-08-31-tpu-coding-research.md) before reusing
the historical conclusions below. Raw timings and JSON have not been changed.

- The depth oracle uses separately jitted JAX blocks, while its hybrid suffix
  uses one JIT. A JAX-only same-suffix control is missing; network amplification
  is a hypothesis, not an established cause of the observed final error.
- The inspected Artgor Q beam minimizes scores. Existing `argmax_agreement`
  is not best-action agreement. Add argmin and actual masked candidate top-K
  comparisons before making search-quality claims.
- Identical saved aggregate error metrics do not prove pairwise-identical
  tensors across fusion boundaries; direct comparisons were not recorded.
- Generated categorical-domain stress inputs are not necessarily reachable
  puzzle states. Both test classes are needed and must be named separately.
- A source-level Dense bias-rounding difference has a CPU witness; its impact
  on compiled TPU results still needs a controlled hardware A/B.
- Hardware generation, physical VMEM, compiler scoped-allocation limits,
  shape-derived FLOP/state and timing-derived dense-equivalent FLOP/s must
  be recorded separately.

## Fixed model contracts

- Artgor checkpoint: `q555_2k_BEST.pt`, 24,757,807 parameters.
- State shape: 150 categorical `uint8` values.
- Embedding shape: `150 x 24`; therefore `NUM_CLASSES=150` and `EMBED_DIM=24`.
- Input dense: `3600 -> 1024`, LayerNorm, ReLU.
- Trunk: 10 residual blocks, each containing two `1024 -> 1024` dense layers
  and two LayerNorm operations.
- Head: `1024 -> MOVE_COUNT`, with `MOVE_COUNT=30` for this checkpoint.
- Inference storage/compute contract under comparison: BF16 model tensors.

## Measured facts

- Original JAX full model at local batch 16,384: approximately 1.39M states/s.
- Standalone Pallas LayerNorm BM128/BM256 is about half the throughput of JAX
  LayerNorm; BM512 and BM1024 exceed 16 MiB VMEM in the tested layout.
- For the embedding model, ordinary embedding gather beat virtual one-hot MXU.
- Per-layer Dense+LayerNorm fusion improved an isolated layer at BM128, but the
  best absolute isolated result was BM256/BK256/BN512.
- Incremental residual-block diagnostic on the same hidden input:
  - Dense1: Pallas/JAX 0.606x; 99.99936% BF16 outputs exact.
  - Dense1+LN+ReLU: 0.394x; 90.566% exact.
  - Two-kernel residual block: 0.283x; 72.146% exact.
- The old `per_layer` implementation was not block fusion: it launched two
  Pallas kernels and materialized the activation between them.

## Corrected mistakes

- `150 x 24` was temporarily misread as 24 categories. Correct meaning is 150
  categories with embedding width 24.
- A repeated `0..149` state is valid but degenerate, not out of domain.
- Full JAX versus fragmented Pallas was not a fair measurement of Pallas's
  block-level potential. It remains useful only as an end-to-end regression.
- Finite output alone is not correctness. Full-model acceptance also requires
  numerical bounds and at least 99% argmax agreement on diverse valid states.

## Active hypotheses

1. Reusing `hidden1`, dense output, and accumulator scratch inside one kernel
   should reduce the large two-kernel residual-block penalty.
2. Current Dense tiling leaves throughput on the table even before LayerNorm;
   BM/BK/BN must be swept rather than inherited from the isolated fusion run.
3. FP32 statistics may improve full-model ranking stability but can reduce
   throughput and differ from the checkpoint's original BF16 execution.
4. A block candidate that wins at batch 4096 may not remain Pareto-optimal at
   16,384 or 32,768, so promotion must remeasure rather than extrapolate.
5. Small per-block BF16 differences may accumulate through ten blocks; block
   correctness does not authorize full-model scaling without the output-head
   argmax gate.

## Comprehensive staged run

1. Screen 32 residual-block candidates at batch 4096:
   BM 128/256, BK 128/256, BN 256/512, BF16/FP32 statistics, and two-kernel
   versus one-kernel boundaries.
2. Promote only the three fastest correctness-valid candidates and remeasure
   them at batches 16,384 and 32,768.
3. Use the winning tiling for full-model `separate`, `per_layer`, and
   `per_block` comparisons against original JAX at batch 16,384.
4. Run 1/8-TPU and 128-chunk scan scaling only after a Pallas full-model
   candidate passes the numerical and argmax gates.
5. Atomically checkpoint JSON after every candidate; compile/VMEM failures are
   recorded and do not terminate the remaining sweep.

## Comprehensive run result: 2026-08-29

- 24/32 block candidates passed; all eight rejections were one-kernel BM256
  scoped-VMEM overflows (16.06-16.36 MiB requested versus 16.00 MiB).
- One-kernel BM128/BK256/BN512 FP32 won batch-4096 screening at 5.295M
  states/s, essentially tied with two-kernel BM256/BK256/BN512 FP32.
- At batch 16,384 and 32,768, two-kernel BM256 won with 7.466M and 8.030M
  states/s; one-kernel BM128 reached 7.230M and 7.481M.
- Full-model JAX reached 1.386M states/s. Pallas separate/per-layer/per-block
  reached 0.577M/0.547M/0.565M with identical 73.69% argmax agreement.
- Because all fusion boundaries produce the same full-model error, arithmetic
  drift accumulates through depth independently of the storage boundary.
- Next diagnostic: per-depth Pallas-prefix/JAX-suffix hybrid curves for hidden
  error and final-head ranking agreement.

## Per-depth diagnostic result: 2026-08-29

- Per-block BM128 and per-layer BM256 produce identical tensors for a fixed
  statistics mode; fusion boundary is not the source of ranking divergence.
- Ranking agreement falls immediately after replacing only residual block 1:
  73.01% with BF16 statistics and 72.14% with FP32 statistics.
- The BF16 block-1 hidden differs by only 0.000545 mean absolute value and has
  cosine 0.999991, but the unchanged JAX suffix amplifies it to 0.172 mean
  output error.
- No later block creates a unique ranking cliff. Block 3 has the largest
  isolated error, while agreement remains roughly 72-74% across depths 1-10.
- FP32 statistics reduce final mean output error at depth 10 from 0.17228 to
  0.14048, but argmax is still only 73.93%.
- Next attribution experiment must independently cross JAX/Pallas Dense and
  JAX/Pallas LayerNorm for block 1, followed by the identical JAX suffix.

## Controlled arithmetic follow-up: 2026-08-31

The historical argmax gates and claims of suffix amplification above are
superseded interpretations, not erased measurements. The consumer minimizes Q;
the depth harness lacked its JAX-only same-suffix control. Matching aggregate
error metrics did not establish tensor equality. See the public 2026-08-31
research audit and expert follow-up for the corrected evidence boundaries.

The [new bundled arithmetic experiment](2026-08-31-layernorm-arithmetic-ab.md)
adds independently tested Dense-before-bias rounding and logical-mean rounding
modes, matched raw/typed runtime JAX baselines, same-suffix controls, real legal
scrambles and minimizing global top-K diagnostics. Legacy defaults and the BN
implementation remain unchanged. CPU regression tests pass; TPU results and
any inference speedup remain pending.

## Arithmetic A/B v1 completed: 2026-08-31

The preceding pending status is superseded by the
[completed report](../../test_results/kaggle_layernorm_arithmetic_v1/report.md)
and raw result at source `2e9602829b8e4fa8498b64461f64c556e77ad4f4`.

- Actual hardware is **TPU v5 lite**, eight visible devices, one active, despite
  a `v3-8` CLI request. Runtime JAX/jaxlib0.10.2, libtpu0.0.42.1 is now recorded.
- Source FP32-parameter JAX still executes BF16. Typed BF16-parameter JAX matches
  it exactly on both corpora at4096/16384; captured source matches at4096.
- JAX-only graph partitioning itself changes Q. At depth1, legal/stress mean
  absolute errors versus monolithic are.148819/.168535 without Pallas. Same-suffix
  self-controls and cross-JAX/JAX are exact. Old perturbation-amplification
  attribution was confounded; this does not retroactively quantify the old run.
- Dense `late` differs in only20/50 and31/59 of4,194,304 first/second Dense
  elements (legal/stress). Explicit BF16-before-bias rounding changes22–26%:
  the CPU expression-level hypothesis does not match the fused TPU execution.
- 52 cases fail with Mosaic boolean-mask relayout errors (8 operators,32 block
  screen,12 full-model); these are not VMEM rejections or measured latencies.
  BF16-statistics LN remains target-unvalidated in this runtime.
- Full JAX is about1.42M states/s at16384. Executed Pallas hybrid/per-block
  cases give about1.00–1.01M/.56M and fail exact-Q acceptance. No32768 promotion,
  scaling run, profile or demonstrated Pallas speedup results from this run.
- Stable top-K is tie-sensitive: legal/stress16K references have5337/1337
  candidates tied at K. Legal walks include duplicate solved states; these are
  not real frontiers, and global top-K remains only a proxy for distributed beam.
- Next proposed tests: isolate aligned-width redundant-mask lowering, match the
  observed JAX mixed-precision LN schedule, and test the still-missing full
  `late Dense + JAX LN` hybrid. No production arithmetic/BN default changed.

## Arithmetic follow-up research: 2026-08-31

[Source/HLO follow-up](2026-08-31-arithmetic-followup-research.md) records new
diagnostic constraints, not a new TPU result:

- The missing full late/JAX candidate changes twenty residual Dense operations;
  input path and Q head remain JAX. Add a matching full cross-JAX/JAX control.
- HLO suggests explicit BF16 mean/variance/invstd boundaries within FP32 vector
  arithmetic; it is not machine-level proof. Three redundant aligned-width
  mask sites are the concrete compiler-reproducer targets.
- JAX 0.10.2 forces layout passes on the TensorCore custom-call route, so a
  requested `needs_layout_passes=False` is not an effective bypass there.
- Full inference is already one compiled host invocation. Microprobe spread
  reaches 16–40% ordinarily, with one 238% range/median outlier; diagnostic
  baseline/candidate profiles are needed independently of acceptance.
- At least 22.04% duplicate rows are forced by zero/one-move strata at 16K.
  This lower bound is not a measured count or permission to weaken the Q gate.
- Pinned JAX maps actual `TPU v5 lite` to v5e; use that generation's hardware
  model, separating physical VMEM from historical scoped allocation limits.

Two project experts reviewed control/consumer questions. Their profile-only-
accepted recommendation was rejected for diagnosis, while acceptance remains
unchanged. No source, BN defaults, kernel launch or automation changed here.

## Arithmetic follow-up implementation: 2026-08-31

User approved the bounded [next bundle](2026-08-31-arithmetic-followup-bundle.md).
The new experimental LN module preserves production defaults and separates
legacy predicate controls, one-mask-at-a-time variants, promoted select
operands, direct2D predicates, and the HLO-informed mixed arithmetic hypothesis.
Full `late Dense + JAX LN` replaces only residual Dense operators; JAX
embedding/input/head and a complete JAX/JAX builder control are retained.

TDD covers expression-level widths130/1024, poisoned padded tails, predicate
ranks, already-compiled timing, retained queue outputs, runtime model controls,
strict promotion, profile eligibility separation, and partial failures.
Initial missing-feature tests failed before implementation. Independent review
found sequential singleton operator timing and an uncaught group-timing failure;
both were corrected with regression tests (five expected failures before fix).
Re-review found no remaining blocking issue. Final local regression:
`python -m pytest -q` ->252 passed in55.70s. This remains CPU/interpreter evidence,
not TPU compilation or acceleration.

Production Dense/LN/block/full measurements now use matched compiled groups.
Failed paired groups retain errors and unpaired diagnostic salvage, which is
ineligible for promotion. Speedups stay null until exactness and comparable
timing hold on both corpora for the same batch. Queued calls are explicitly
not real128-chunk scans; profiles are diagnostic even for rejected candidates.
Launch provenance and actual TPU results will be recorded separately.
