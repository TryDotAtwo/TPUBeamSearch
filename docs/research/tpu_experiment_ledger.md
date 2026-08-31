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

## Arithmetic follow-up v1 submitted: 2026-08-31

Public source `d58cf9fd8e86ec145c6bbc4f6c7f5aff489d6e21` and launcher
`d87fa2b16d5bc3489d914939db0ce4ba7766b397` were pushed before Kaggle submission.
Private `trydotatwo/tpu-layernorm-arithmetic-followup` accepted version1 and
reported RUNNING at06:22:37 MSK. Actual device/runtime and execution results
are not yet verified. See [launch record](../../test_results/kaggle_layernorm_followup_v1/launch.md).
Heartbeat `check-tpu-arithmetic-follow-up` checks every10minutes and removes
itself after terminal analysis/publication. No previous kernel was restarted.

## Arithmetic follow-up v1 completed: 2026-08-31

The pending launch status above is superseded by the
[completed report](../../test_results/kaggle_layernorm_followup_v1/report.md)
at source `d58cf9fd8e86ec145c6bbc4f6c7f5aff489d6e21`.

- Runtime remains JAX/jaxlib0.10.2, libtpu0.0.42.1, actual **TPU v5 lite**, eight
  visible / one active. Checkpoint, original-model-source, puzzle and both
  32768-input hashes match arithmetic v1. All258 output paths plus the full
  Kaggle log were downloaded; two oversized identical StableHLO files are
  published losslessly compressed, with raw-byte hashes in the manifest.
- Of56 synthetic and36 checkpoint-operator cases,14+10 fail Mosaic predicate
  relayout compilation, not VMEM allocation. All14 block,14 full,6 baseline
  and2 same-suffix-control rows execute. All8 paired timing groups succeed.
- Minimal BF16 rank1-predicate broadcast fails4/4 width/BM cases; direct2D
  succeeds4/4 exactly. FP32 selection succeeds with either construction.
  Removing aligned redundant masks and FP32-where also permit LN compilation.
  These are measured compiler workarounds, not arithmetic equivalence.
- Mixed LN improves standalone exact fractions from about48%/53% to90%/88%
  (legal/stress), but no surviving LN operator is exact. The exact remaining
  target arithmetic mechanism is unresolved; matching coarse HLO dtype
  boundaries is insufficient.
- Late Dense differs in20/31 of4,194,304 standalone elements, yet full late/JAX
  Q exactness is32.87%/17.59%, argmin agreement92.65%/86.74%, topK overlap
  75.06%/92.99%. Same-suffix JAX-only controls separate a real monolithic-versus-
  partitioned-JAX effect; aggregate errors must not be subtracted to attribute it.
- All six Pallas-containing full configurations fail exact Q on each16K corpus
  and lose to original JAX in all12 paired rounds. Original11.494/11.504ms;
  late/JAX16.068/16.088ms. Typed/captured/JAX-builder controls are exact. All
  eligible speedups remain null: **no32K promotion or8TPU scaling**.
- Ten device Chrome traces show20 replaced residual operators per forward.
  FP32-where LN accounts for about14.36ms extra module time versus unmasked;
  mixed masked LN for about8.83ms. Queued same-executable calls reduce observed
  full-call cost by0.33–0.48ms, not those device penalties; they are not real
  128-chunk scans. XPlane files are retained but not decoded.
- Common embedding gather plus flattening reshape costs about5.47ms of the
  original10.97ms device module. Next proposed target: exact-value flat gather,
  isolated and inside full JAX with unchanged Q gate. No speedup is inferred
  from operation removal, and no new TPU job is launched by this analysis.
- Legal16K has11401 unique states /11606(state,last_move) pairs; duplicate
  representatives are30.41%. K-boundary ties remain numerous. Stress inverse
  masking is a no-op because last_move=-1. Global topK is not distributed beam.

No BN or production inference default changed. Fresh local regression:
252 tests passed; reproducible numerical and device-profile summaries are
published alongside raw artifacts. The completed follow-up monitor is retired
after publication, without submitting a new TPU job.

## JAX/Pallas execution attribution: 2026-08-31

[Detailed source/profile audit](2026-08-31-jax-pallas-execution-attribution.md)
adds attribution from the existing follow-up v1, not a new TPU measurement:

- Twenty JAX residual Dense-containing operations sum to3.8546ms, versus
  8.3370ms for twenty late Pallas Dense calls. JAX fuses bias, following LN sum
  and often preceding vector work; these are not matched isolated GEMM timings.
  The device gap is not explained by Python dispatch.
- Typed JAX outer HLO branches the FP32 biased result into the LN sum and a
  separate BF16 output conversion; the Pallas boundary returns BF16 before the
  sum. BF16 type-correction metadata prevents asserting final rounding behavior
  from this alone. Matched JAX barriers and intermediate witnesses are proposed.
- Current Dense tiles are128/256/512 with exactly aligned residual dimensions.
  Different layouts and scoped VMEM windows are observed, not their individual
  causal costs. Default Pallas pipelining already double-buffers; literal
  HBM charging for all tile-window reads contradicts observed latency.
- The centered-value predicate is the strongest isolated mixed-LN mask cost;
  full mixed-direct2D remains unmeasured. Faster LN arms still fail exactness.
- Correct the old "Pallas embedding gather" interpretation: it was ordinary
  JAX gather followed by Pallas Dense/LN, not a custom gather kernel. Existing
  compiled JAX already casts the embedding table before gather. A true flat
  gather remains a proposed exact-value optimization, shared-input cost rather
  than an explanation of the candidate gap.

Three independent audits and two project experts informed the controls; raw
source/HLO/traces determine claims. No BN/default changes, new kernel launch,
accepted Pallas result or multi-device scaling are implied. The next bounded
bundle awaits agreement on its proposed scope.

## Execution-boundary bundle implementation: 2026-08-31

The user approved the proposed bundle, then explicitly requested no subagents.
The approval-pending sentence above is superseded by this
[frozen protocol](2026-08-31-execution-boundary-bundle.md).

New opt-in code implements nine matched Dense/boundary arms, five LN arms,
five embedding arms, direct numerical witnesses and observed JAX statistics;
15 full configurations preserve the original checkpoint and exact-Q gate.
Only eligible non-controls advance from both16K corpora to actual32K.
No BN/default path changes. No target speedup is claimed by implementation.

Local TDD found and removed unsupported general/CLIP gathers in the new banked
embedding. The replacement matches the pinned Mosaic gather source contract;
target TPU layout and compilation still need measurement. Added tests cover
classes128..149, output/state tails, runtime parameter sensitivity, direct
comparison witnesses, strict JSON, genuine larger-batch promotion, partial
failure artifacts and provenance. Full regression:281 passed in102.81s.

Kaggle submission and runtime results are separate from these local checks.
The launch record must state whether the free-slot check and actual submission
succeeded; source publication alone is not a queued or running experiment.

## Execution-boundary v1 submitted: 2026-08-31

[Launch record](../../test_results/kaggle_execution_boundary_v1/launch.md)
supersedes the earlier403/preflight blocker. Existing project TPU notebooks
were terminal (19 COMPLETE,1 old ERROR); two additional recent owned jobs were
also terminal. Transient TLS/proxy failures were resolved with read-only retries,
without changing networking or credentials.

Private `trydotatwo/tpu-execution-boundary-ab` version1 was submitted once from
launcher `51c8b3a512e650df83939d322c78bd715cfd8221`, pinned to public source
`45062324d368f4849adb6d572d21d54f75854d79`. Live QUEUED confirmed17:14:09 UTC.
Downloaded server bootstrap matches the launcher after newline normalization;
private/TPU/GPU metadata and source SHA were verified. Requested `v3-8` maps to
server `TpuV5E8`; runtime inventory and numerical/performance results remain
unverified. No BN/default or inference-source changes. The existing heartbeat
now monitors this submitted version without resubmitting it.
