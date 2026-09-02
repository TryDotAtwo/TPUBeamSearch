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

## Execution-boundary v1 completed: 2026-08-31

[Terminal report](../../test_results/kaggle_execution_boundary_v1/report.md)
supersedes the queued record. Source remains
`45062324d368f4849adb6d572d21d54f75854d79`; actualTPUv5lite,8visible/1active,
JAX/jaxlib0.10.2 and libtpu0.0.42.1. All recorded checkpoint/model-source/puzzle/
input hashes match follow-up v1. All106 cases execute; zero compile or case
errors. All12 paired timing groups succeed. All240 output files plus full
Kaggle log are downloaded and hashed; no signed download URLs are published.

- **Pallas banked embedding + otherwise JAX model passes finite elementwise
  exact original Q on both16K and both actually executed32K corpora.** Full
 16K takes9.151/9.134ms versus original11.593/11.677ms,1.267/1.278x throughput.
 32K takes18.549/18.626ms versus24.122/24.138ms,1.300/1.296x. It also beats
 typedBF16 JAX and the exact tiled-JAX lookup in every paired round.
- Tiled JAX is also exact and gives1.136/1.156x at16K,1.153/1.150x at32K.
 Naive flat JAX is exact but takes about570ms at16K and is not promoted.
 Isolated4K lookup favors tiled JAX; full-graph evidence determines the winner.
- Device modules average10.967ms original,10.942ms typed,9.418ms tiled,
 8.564ms banked. Banked lookup is2.414ms, with runtime table-preparation
 loops still timed. Original gather+flatten costs5.474ms. Static temp storage
 drops723.660→123.967MiB at16K; these are allocations, not bandwidth counters.
- Pallas Dense BM128→512 reduces20-call device cost8.337→5.216ms. BK1024
 reaches exact standalone Dense and same-compiled-LN output on both4K samples,
 but full Q still fails. All-JAX separate Dense/LN and JAX post-barrier controls
 already differ from composed JAX. HLO shows different sum/output boundaries
 and emitters; outerFP32 types are not complete machine-rounding proof.
- Mixed LN/direct2D remain inexact versus JAX. Direct2D and unmasked outputs
 match in direct4K controls and show the same full aggregate errors, essentially
 equal device cost. FP32-where is slower. All Dense/LN full interventions fail
 unchanged exact-Q gate; high cosine or topK overlap does not override it.
- Profile analysis accounts for inclusive while/body spans without double
 counting.16/17 traces pass strict checks; flat-JAX's5.957us module-clock
 discrepancy is retained as a diagnostic rejection, not silently corrected.
 XPlane files are retained but not decoded. Unprofiled timing is separate.
- Exact variants preserve minimizing topK identities/order, masks and ties.
 Legal inputs include duplicates; globaltopK is not distributed beam. Queued
 eight-call diagnostics are not real128-chunk scan. No8-device scaling yet.

Analysis and regression work ran inline without subagents. Reproducible raw-
cell CSV validation, artifact hashing and profile checks accompany the report.
Fresh full regression:295 passed in89.23s; deterministic analysis check passed.
No inference-source, BN or production-default changes. Next proposed target is
the exact hybrid's real caller/scaling; no new kernel is launched here. The
completed execution-boundary monitor is retired after publication.

## Exact eight-device inference v1: 2026-09-01

[Terminal report](../../test_results/kaggle_inference_8device_v1/report.md)
records private kernel `trydotatwo/tpu-exact-inference-8-device` v1 from public
source `d2159cb230ef77deeb5a4a2b6a42181a62dc027c`: JAX/jaxlib0.10.2,
libtpu0.0.42.1 and eight real TPU v5 lite devices in one process.

- Prepacked FP32-bank Pallas BM2048 reaches17.413/17.921M states/s at fixed
  local batch16,384, versus10.771/10.737M for original JAX:1.617/1.669x.
  On the first one-device16K prefix it reaches2.459/2.454M states/s and is
  elementwise exact.
- Promotion is rejected.  At global batch131,072 it differs in12 legal and55
  stress Q values; stress changes one argmin.  No32K confirmation or profiles
  run after that frozen gate failure.
- Tiled JAX, runtime-bank Pallas and all successful FP32 prepacked tile sizes
  share the same candidate output hash.  Typed JAX exactly matches original.
  First-Dense HLO layouts/emitter/window configuration also match between typed
  and tiled JAX.  Of22 MXU convolutions, only the final residual Dense changes
  schedule: typed JAX uses iteration bounds2x16x1 while every fast encoding
  graph uses1x22x1.  This appears on both1-device and8-device HLO; it is a
  strong state-sensitive rounding hypothesis, not yet causal proof.
- The old one-device screen did not contain global witness rows29,807,50,224
  or29,369.  The result therefore does not yet prove an eight-core cause.
- BF16 physical banks fail all12 compile attempts because Mosaic does not
  implement the observed different-bitwidth dynamic gather.  FP32 banks compile
  after the same logical BF16 rounding.
- Static temp estimates fall from about723.66MiB original to123.88MiB banked;
  these values are not hardware counters.

The next frozen execution bundle is documented in
[Exact eight-TPU inference execution A/B](2026-09-01-inference-execution-ab.md)
and published at source `88d6e42c4100578aa9478d3faf6b4f5d30adc01f`.
It replays exact witness-owner shards on one device and compares input
boundaries, split dispatch, direct sharded jit, pmap and independent one-core
executables.  Inference only; BN/defaults and beam-search stages remain
unchanged.

## Exact eight-device execution A/B v1: 2026-09-01

[Terminal report](../../test_results/kaggle_inference_execution_ab_v1/report.md)
records private `trydotatwo/tpu-inference-execution-ab` v1 at source
`88d6e42c4100578aa9478d3faf6b4f5d30adc01f`.  The runtime again exposes eight
active TPU v5 lite devices; all checkpoint/model/input hashes match the previous
run.

- No arm passes exact-Q promotion.  Fastest exact typed JAX takes
  11.881/11.968 ms on legal/stress at local batch 16K.  Pallas banked encoding
  takes 7.181/7.312 ms, a rejected 1.654/1.637x opportunity with 12/55 Q
  mismatches.  No 32K confirmation runs.
- Exact witness-owner 16K shards replay the same drift on one TPU.  Eight-device
  count is therefore not causal; the earlier one-device prefix omitted the
  witnesses.
- Original `shard_map`, explicit sharded `jit`, `pmap` and eight independent
  executables share exact output hashes.  Pallas under `shard_map`, `pmap` and
  independent executables also shares one common rejected hash.  Launch API and
  host orchestration do not explain the arithmetic.
- JAX and Pallas two-dispatch encoding splits share an identical rejected hash
  (93/427 mismatches).  Hence the Pallas encoded BF16 tensor is exact and the
  downstream executable schedule changes when the graph is cut.
- Returning every internal boundary makes typed JAX and Pallas elementwise
  identical through all ten blocks and Q on every witness.  Observability
  perturbs fusion; the unobserved full replay remains the correctness evidence.
- Input pre-barriers do nothing; post-input-Dense barriers cause large drift.
  The next experiment targets the already identified final residual Dense
  schedule with block-9/Dense2 barriers and materialized execution cuts.

Raw JSON, full logs and 146 StableHLO/compiled-HLO files are retained.  Two
expected direct-Pallas compile failures are recorded, not hidden.  No BN or
production-default path changes.

## Final-residual A/B prepared: 2026-09-01

The next [frozen protocol](2026-09-01-final-residual-ab.md) follows directly
from execution A/B v1.  New opt-in builders preserve all model formulas while
testing final-block barriers, one-dispatch output taps and device-resident cuts
before/after the final block's second Dense.  Every tap and split has a
structurally matched typed-JAX control.  Promotion remains exact full BF16 Q on
both corpora and a speed win over the fastest exact JAX at real local batches
16K and 32K across eight devices.  No BN/default or beam-search path changes.

## Exact eight-device inference target achieved: 2026-09-01

[Terminal report](../../test_results/kaggle_final_residual_ab_v1/report.md)
records private `trydotatwo/tpu-final-residual-ab` v1 at public benchmark source
`267df37cd3a35b19ad6250d43768bfd5b536b67c`.  The runtime is Python3.12.13,
JAX/jaxlib0.10.2, libtpu0.0.42.1 and eight active TPUv5lite devices in one
process.  All checkpoint/model/puzzle/input hashes match the preceding
execution A/B; all54 measurements complete and `error_count=0`.

- **`pallas_split_after_final_block` passes the frozen goal.**  It performs the
  prepacked banked Pallas embedding and unchanged JAX ResMLP through block9 in
  one dispatch, keeps the BF16 hidden matrix device-resident, then applies the
  JAX head in a second dispatch.
- At real local batch16,384 it takes7.321/7.312ms on legal/stress versus
  11.921/11.929ms for the fastest exact JAX controls:1.628/1.631x and
  17.904/17.925M global states/s.  At real local batch32,768 it confirms at
  15.647/15.653ms versus24.769/24.872ms:1.583/1.589x and16.753/16.747M/s.
- Full BF16 Q is finite and elementwise exact on all four runs: zero mismatches,
  zero max/mean/RMSE and argmin agreement1.0.  The same winner is selected at
  32K; no threshold is weakened.
- Of22 monolithic MXU Dense schedules, only final residual Dense2 differed:
  exact typed JAX used2x16x1 while the fast inexact Pallas monolith used1x22x1.
  All five barrier arms retain1x22x1 and fail.  A tap or real split after the
  complete final block restores2x16x1 and exact Q; the separate head compiles as
  one1x3x1 MXU operation.  Cuts inside the block change BF16 materialization
  and are rejected.
- Compiler static temporary estimates are758,779,904B for typed monolithic and
  129,827,840B for the winner prefix; the latter materializes a33,554,432B
  local hidden output and its head reports zero temporary bytes.  These are
  allocation estimates, not memory-traffic counters.

The measured formulas are promoted as the opt-in reusable
`tpu_beam_search.stream1_layernorm_exact` two-stage API.  Its host-level call
must not be enclosed in another outer `jax.jit`, because the dispatch boundary
is part of the validated compiled program.  No BN/default path or beam-search
stage changes.  Fresh local regression:337 tests passed in131.44s.

## Exact inference frontier v1 completed: 2026-09-01

[Terminal report](../../test_results/kaggle_exact_inference_frontier_v1/report.md)
records private `trydotatwo/tpu-exact-inference-frontier` v1 at benchmark
source `fc5c87ae5c49c0a92d4ccd634831e8980a7f44e8`. Runtime is Python3.12.13,
JAX/jaxlib0.10.2, libtpu0.0.42.1 and eight active TPUv5lite devices in one
process. Checkpoint/model/puzzle/input hashes match the frozen protocol.

- The confirmed winner is
  `exact_split_bm4096_pallas_head_bm256_bk1024_bn128_late`: exact Pallas
  banked embedding plus unchanged JAX/XLA residual prefix at BM4096, followed
  by a device-resident Pallas head at BM256/BK1024/BN128 with late rounding.
- At local batch16,384 it takes7.133/7.146ms legal/stress versus11.983/11.903ms
  original JAX and7.284/7.230ms accepted BM2048+JAX-head control. At local
  batch32,768 it confirms at15.363/15.364ms versus24.865/24.852ms original and
  15.566/15.541ms accepted control. This is17.063M global states/s and
  1.618/1.618x versus original at confirmation.
- All four selected outputs are finite, elementwise and hash exact: zero
  max/mean/RMSE and mismatch witnesses. The screen winner remains the 32K
  winner under the unchanged gate.
- Exact head arithmetic is boundary-sensitive. Late BK128/BK1024 are exact;
  late BK256/BK512 and every pre-bias-rounding arm are rejected. Three
  materialized-identity arms also produce427/663 legal/stress mismatches.
- Prefix BM8192/BM16384 are compile rejections at19.84/24.25MiB against the
 16MiB scoped VMEM limit. At32K, compiler static temp estimates fall from
  about1.510GB monolithic to251,946,496B for the prefix; these are allocations,
  not traffic counters.
- The Pallas head's marginal composed improvement over the same BM4096 prefix
  with a JAX head is only about0.14/0.33%, while standalone Pallas head is
  slower. Do not attribute the robust full-model gain to the head alone; the
  exact split and prefix tiling dominate.
- Full JSON/log and110 HLO files are retained. A Kaggle backend inconsistency
  returned `kernels.get`403 for the completed newest private kernel despite
  valid quota, owner listing and terminal logs. One of16 diagnostic
  trace/XPlane pairs was recovered; its two-dispatch accepted control passes
  strict TPU0 analysis. Missing winner traces are disclosed, not reconstructed.

This advances the inference frontier but remains a hybrid: the ten residual
blocks are JAX/XLA-lowered. An all-Pallas replacement still needs exact
per-boundary arithmetic and a full32K/device win before promotion. No BN,
default or beam-search path changes. Post-download verification: six focused
artifact tests, twenty plugin-package tests and345 full-project tests pass.

## Exact Artgor notebook publication gate passed: 2026-09-01

[Terminal report](../../test_results/artgor_exact_notebook_validation_v2/report.md)
records private `trydotatwo/tpu-artgor-exact-notebook-validation` v2 at source
`2b99bdf5116f828a21d35b2c5910467f6ab039c2`.  Runtime is Python3.12.13,
JAX/jaxlib0.10.2, libtpu0.0.42.1 and eight TPUv5lite devices with x64 enabled.

- Full-Q inference at32,768 states/device is bitwise exact on legal and stress
  corpora.  It takes15.515/15.289ms versus24.636/24.787ms original JAX:
  1.588/1.621x and16.896/17.146M global states/s.
- One depth has all13 output tensors hash exact.  Three consecutive depths have
  all39 tensors, frontiers and backpointers exact; their steady paired depth
  speedup is1.103x at the smaller1,048,576 global parity beam.
- The real16,777,216-beam gate finds pid1034 at depth110; its116-move path
  independently replays to solved.  Solver wall time is2,955.75s.  There is no
  paired original full solve, so this does not establish a full-solver speedup.
- Independent download validation re-derived all nine gates, checked54 primitive
  comparison records, matched private input hashes and replayed the path.  The
  public report omits competition rows, states and the solution itself.
- V1 failed because x64 promoted Python zero literals in the banked Pallas LUT
  `BlockSpec`, yielding illegal `(i32,i64,i64)` Mosaic indices.  Commit
  `2b99bdf` fixes all LUT indices to int32;371 local tests pass before v2.

## Packaged Artgor notebook v2 completed: 2026-09-02

[Safe report](../../test_results/artgor_exact_public_notebook_v2/report.md)
records the actual packaged notebook v2 at runtime source `2b99bdf`.  Its four
frame-runs exactly match the preserved Artgor scriptVersionId344319112 run on
pid/frame/inversion/found status, beam, checkpoint, found path length and path
hash.  Both found paths report runtime `verify=True`.

The matched four-record total falls from25,518.988s to21,928.830s, a measured
**1.1637x** full-frame wall-speedup.  Per-frame ratios are1.1598x-1.1665x.
This is the comparable end-to-end solver evidence; it remains distinct from
the larger model-only inference speedup.

## Strict exact-split speed proof passed: 2026-09-02

[Terminal report](../../test_results/artgor_exact_speed_proof_v1/report.md)
records private `trydotatwo/tpu-artgor-exact-speed-proof` v1 at source
`3070839d4f04cff8fa58794024384c9bd98aa947`.  It uses eight TPUv5lite devices,
32,768 states/device, three warmups and21 alternating synchronized samples for
each of three legal and three categorical-stress seeds.

- All six full-Q outputs are BF16 hash exact with zero mismatches.
- All126 paired observations exceed the frozen1.5x threshold.
- The weakest case median is1.5836x, weakest individual pair1.5219x and
  weakest one-sided bootstrap lower99 bound1.5755x.
- The downloaded raw pairs were independently recomputed with the frozen
  statistic implementation and per-case bootstrap seeds; every gate passes.

This establishes the component claim: exact-split full-Q inference is at least
1.5x faster than unchanged JAX under the frozen protocol.  It does not promote
the whole solver to a1.5x claim.  The all-Pallas44-boundary diagnostic is the
next sequential TPU experiment; exact-split remains the production fallback.

## All-Pallas transparent diagnostic v3: 2026-09-02

[Terminal report](../../test_results/artgor_pallas_exact_diagnostic_v3/report.md)
records private `trydotatwo/tpu-artgor-all-pallas-exact-diagnostic` v3 at source
`7888c0e548d111f53861d766193f41bee58df81a` on eight TPUv5lite devices.  V1 and
v2 were compile-only failures (illegal rank-one BN128 bias tile, then mixed
i32/i64 LayerNorm BlockSpec indices); v3 is the first arithmetic result.

- Prepacked Pallas embedding is BF16 exact for both candidates on all six
  legal/stress corpora.
- BK128 input Dense is BF16 exact on all six corpora.  The first mismatch is
  always `input.layernorm_relu`; legal42 differs in163,288/2,097,152 elements
  (max abs about0.03), stress43 in154,177 (max abs about0.02).
- BK1024 first diverges in input Dense (120 legal42 and174 stress43 elements),
  so it is retained only as a reduction-order negative control.
- Neither candidate reaches performance promotion.  No fusion or all-Pallas
  speed result is claimed; exact-split remains the production engine.

The next causal bundle holds the exact BK128 Dense input fixed and attributes
LayerNorm in observable stages: JAX monolithic/decomposed/materialized controls
and Pallas mean, centered, variance, rsqrt, affine and ReLU checkpoints.  It
uses one-factor changes rather than a Cartesian dtype sweep.

## LayerNorm arithmetic attribution v4: 2026-09-02

[Terminal report](../../test_results/artgor_layernorm_attribution_v4/report.md)
records private `trydotatwo/tpu-artgor-layernorm-arithmetic-attribution` v4 at
source `fee05807dc82a836bf3c8c17aba03131033b86c9`, eight TPUv5lite devices and
256 rows/device across all six frozen corpora.

- Baseline Pallas mean is exact against decomposed same-call JAX on all cases.
- First Pallas drift is centered FP32 subtraction: 2,087,923--2,096,937 of
  2,097,152 elements, max abs 0.000244--0.000977.  BF16 variance later rounds
  back to exact, but final ReLU still differs in445,631--550,725 elements.
- Monolithic, same-call decomposed and separately materialized JAX controls
  also disagree.  Original versus same-call differs in424,968--520,663 final
  elements; original versus materialized in132,523--175,840.  The unchanged
  monolithic model remains the oracle; the controls expose boundary-sensitive
  lowering rather than authorizing a semantics change.
- No candidate is promoted or timed.  The next causal probe freezes the exact
  BF16 values and mean and compares subtraction-only Pallas, interpret Pallas,
  same-call JAX and materialized JAX before one centered-to-variance fusion
  control.  It must also record RMSE, hashes and module identity omitted by v4.

## Fixed-operand centered subtraction v1: 2026-09-02

[Terminal report](../../test_results/artgor_layernorm_subtraction_v1/report.md)
records private `trydotatwo/tpu-artgor-layernorm-subtraction` v1 at source
`a5b7690fd0e3b24a98e26fe2c134b93308107762`, eight TPUv5lite devices and all
six frozen B256/device corpora.

- Fixed BF16 Dense values and exact BF16 mean produce bitwise-identical FP32
  centered tensors in same-call JAX, materialized-cast JAX, Pallas interpret
  and real standalone Pallas: zero mismatches across every corpus.
- A one-custom-call Pallas centered-to-BF16-variance kernel is also exact
  against materialized JAX variance on every corpus.
- StableHLO identities are distinct and both real Pallas arms contain one
  `tpu_custom_call`; this is real TPU evidence, not an interpret-only result.

Therefore the earlier drift is caused by keeping mean production and centered
subtraction inside the larger LayerNorm Pallas kernel, not by an unavoidable
Pallas FP32 subtraction precision limit.  The next correctness baseline splits
LayerNorm at the mean materialization boundary; dispatch reduction or an
in-kernel VMEM barrier is deferred until full-model exactness is recovered.

## Split-mean all-Pallas diagnostic v4: 2026-09-02

[Terminal report](../../test_results/artgor_pallas_exact_diagnostic_v4/report.md)
records private all-Pallas diagnostic v4 at source `8d9ce0a` on eight TPUv5lite
devices.  Explicit BF16 mean materialization does not change the final
LayerNorm result: all six `input.layernorm_relu` mismatch counts and errors are
identical to unsplit v3 (154,177--194,400 mismatches, max abs0.015625--0.03125).

Together with the exact subtraction-only result, this shows centered FP32 drift
is rounded away at BF16 variance.  The causal investigation advances to the
exact variance feeding epsilon/add/rsqrt/BF16 invstd and then affine.  No timing
or fusion result is claimed.

## Fixed-variance invstd and affine v1: 2026-09-02

[Terminal report](../../test_results/artgor_layernorm_invstd_v1/report.md)
records private `trydotatwo/tpu-artgor-layernorm-invstd` v1 at source
`df562624015a2b27b722e14915138d7345c0764b`, eight TPUv5lite devices and all six
B256/device corpora.

Same-call JAX, materialized JAX, Pallas interpret and real Pallas FP32 invstd
are hash exact.  BF16-rounded invstd is also exact, and one-custom-call Pallas
affine matches JAX exactly from explicit centered/invstd/scale/bias inputs.
All error metrics are zero.  Consequently neither rsqrt nor affine is an
intrinsic Pallas precision limitation; drift arises from fused producer/
consumer lifetime inside the larger LayerNorm kernel.  The next correctness
baseline materializes mean, variance and BF16 invstd between Pallas calls.

## Fully materialized all-Pallas diagnostic v5: 2026-09-02

[Terminal report](../../test_results/artgor_pallas_exact_diagnostic_v5/report.md)
records private all-Pallas diagnostic v5 at source
`c642863f3fee4e1e2ae170a239245a2dae54097b` on eight TPUv5lite devices.

- Embedding and BK128 input Dense are bitwise exact on all six frozen corpora.
- Five explicit Pallas calls per LayerNorm still first diverge at
  `input.layernorm_relu`: 154,177--194,400 BF16 values differ, with max abs
  0.015625--0.03125.
- These counts are identical to the unsplit and split-mean all-Pallas runs.
  Since the fixed-operand mean/subtraction/variance/invstd/affine probes are
  exact, the next diagnostic compares the same modular Pallas intermediates
  with monolithic and explicitly materialized JAX controls. Timing remains
  blocked by correctness.

## LayerNorm boundary replay v1: 2026-09-02

[Terminal report](../../test_results/artgor_layernorm_boundary_replay_v1/report.md)
records private boundary replay v1 at source
`5ee2b43719addd5c2e205f61039e2f6ddd07274c` on eight TPUv5lite devices.

- StableHLO contains the expected five `tpu_custom_call` operations.
- Real modular Pallas is hash exact with separately materialized JAX at mean,
  centered FP32, BF16 variance, BF16 invstd and affine+ReLU on all six frozen
  B256/device corpora. Every boundary has zero mismatches and zero error.
- The monolithic Artgor JAX final differs from both exact-matching modular
  paths by 132,523--175,840 BF16 elements (max abs0.015625--0.03125,
  RMSE0.000663--0.000728). Pallas and materialized JAX final hashes are equal
  in every case.

Thus the remaining target is the effective arithmetic of the unmaterialized
monolithic JAX lowering, not an isolated Pallas primitive. The next one-factor
ladder compares final hashes directly against that oracle before any timing or
fusion optimization.

## Monolithic LayerNorm arithmetic match v1: 2026-09-02

[Terminal report](../../test_results/artgor_layernorm_monolithic_match_v1/report.md)
records private arithmetic match v1 at source
`a50f6490abc6e65428d73a86ab9ae1122ace28d3` on eight TPUv5lite devices.

- The real one-kernel Pallas `fp32_variance` arm is hash exact against the
  unchanged monolithic Artgor JAX LayerNorm on all six frozen corpora: zero
  BF16 mismatches in every case.
- All other Pallas arms fail. The materialized-style baseline differs by
  132,523--175,840 elements.
- No explicit JAX one-factor arm is exact because its separate JIT boundary
  changes the effective lowering. The direct Pallas-versus-monolithic equality
  is the promotion evidence.

The promoted one-call contract uses BF16-rounded mean, FP32 centered values and
variance, BF16-rounded invstd, FP32 affine, final BF16 rounding, optional skip
and ReLU. It advances to the full44-stage all-Pallas gate before timing.

## FP32-variance all-Pallas diagnostic v6: 2026-09-02

[Terminal report](../../test_results/artgor_pallas_exact_diagnostic_v6/report.md)
records private full-model diagnostic v6 at source
`989715789d8eff007fb08c143ccedcfeb8121e27` on eight TPUv5lite devices.

- Embedding and BK128 input Dense are exact.
- First mismatch is still `input.layernorm_relu`, reduced to
  21,165--25,312 BF16 elements (max abs0.0078125--0.015625).
- The standalone one-call arithmetic-identical Pallas probe is hash exact.
  The production-only difference is an unconditional column predicate/select
  path even at aligned width1024; the probe omits it statically.

V7 removes masking for aligned widths, preserving it only for real padding.
This simultaneously removes avoidable vector work and tests whether the
predicate changed Mosaic arithmetic lowering. Timing remains correctness-gated.
