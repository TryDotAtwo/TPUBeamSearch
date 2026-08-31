# Execution boundaries, LayerNorm and flat embedding A/B

Implementation approved on2026-08-31 after the
[execution attribution audit](2026-08-31-jax-pallas-execution-attribution.md).
All implementation and final review run inline, following the user's explicit
request not to use subagents. This document freezes the protocol before TPU
measurements. It does not claim a new performance or numerical result.

## Invariants

- Artgor Q ResMLP: state150, categories150, embedding24, hidden1024, ten
  residual blocks,30 minimizing-Q outputs. Embedding150x24 is not24 classes.
- The original monolithic model receives runtime FP32 parameters; the typed
  JAX control receives BF16 parameters. Candidates retain runtime FP32
  embedding, converted before lookup, with BF16 remaining parameters.
  No captured constants or hidden embedding precomputation are timed as runtime.
- Reuse follow-up v1 checkpoint, original Python model, puzzle and both32768
  input hashes. Mismatch is a fatal preflight error. Legal walks seed42;
  categorical stress seed43. Inputs remain unchanged, not deduplicated or
  represented as real beam frontiers.
- The full gate remains finite, elementwise exact monolithic Q on both16384
  corpora. RMSE, cosine, max/mean abs, exact fraction, minimizing argmin,
  global top-K overlap/order, masks and ties are retained. Argmax is not the
  decision criterion. Global top-K is a proxy, not distributed beam selection.
- At most two eligible non-control candidates advance to an actual32768
  execution on both corpora. Eligibility is not proof of being faster; report
  latency distributions and ratios, including ratios below1, separately.
- No BN path, production defaults or existing Pallas Dense/LN implementation
  is changed. New code is opt-in.

## Experimental matrix

On the same4096-state hidden input, independently for legal and stress:

| Group | Arms per corpus | Question |
|---|---:|---|
| Dense |9| JAX boundaries vs Pallas late-rounding tiles |
| Dense+JAX LN |9| Cost and arithmetic changes at the consumer boundary |
| LN only |5| JAX, legacy unmasked, mixed unmasked/direct2D/FP32-where |
| Embedding only |5| Runtime original, typed original, flat JAX, tiled JAX, banked Pallas |
| Instrumented JAX observations |2| LN and Dense+LN mean/variance/inverse/output |
| Direct controls |2| Separate Dense/LN vs composed JAX; mixed none vs direct2D |

Thus56 timed operator rows,4 observations and up to4 direct controls.
Full16384 adds30 configuration rows and4 baseline rows. Nominal total98
rows before conditional confirmation. Individual compilation failures remain
explicit rows rather than silently reducing the declared matrix.

Dense arms are JAX none/pre/post/both optimization barriers, then late Pallas
BM128/256/512 at BK256,BN512; BN1024 at BM128,BK256; and BK1024 at
BM128,BN512. The last arm changes reduction scheduling as well as memory
tiling. A JAX optimization barrier is a graph control, not a guarantee of a
separate TPU dispatch, HBM roundtrip or preserved machine arithmetic.
Compiled HLO and traces must establish the resulting boundary.

The15 full configurations are the nine Dense arms with JAX LN, three JAX-Dense
LN variants (legacy unmasked, mixed unmasked, mixed direct2D), and three flat
embedding arms with the otherwise JAX model. Dense/LN interventions replace
all20 residual layers; input Dense/LN and head stay JAX. Embedding arms change
only the lookup. Independent arms are not combined before measurements.

## Flat embedding contract

All implementations return position-major BF16 `[B,STATE_LEN*EMBED_DIM]`:
for this checkpoint `[B,3600]`. No folding of embedding into Dense or modified
contraction order is introduced.

- `jax_flat`: index a flat table directly with position/category coordinates.
- `jax_tiled`: ordinary lookup/flatten per row tile through `lax.map`.
- `pallas_banked`: output tiles BMx128,128-position state banks, two128-category
  lookup banks, FP32 selection and BF16 output. The phase-dependent lookup
  layout is built from runtime parameters inside the compiled/timed call.
  For embedding24, three phases cover the128-column tile offsets. Tail3600
  is masked, row/state storage is padded and sliced, classes128..149 remain
  unsigned. Unsupported category/embedding geometry is rejected explicitly.

An interpreter-passing first draft used general/CLIP gather, which is not
accepted by the pinned
[JAX0.10.2 Mosaic gather lowering](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/pallas/mosaic/lowering.py).
A failing regression exposed the unsupported gather contract; the kernel now
uses batched `take_along_axis` with mathematically bounded indices and
`promise_in_bounds`. JAXPR regression checks all three gathers. This removes
a known source-level incompatibility; actual TPU layout/compilation remains
unverified until the run. CPU interpretation cannot prove TPU legality.

## Attribution and measurement

- Dense outputs are compared directly, with up to16 differing coordinates,
  values, dtypes and byte witnesses. Each Dense output also feeds exactly the
  same separately compiled LN executable. Never subtract aggregate errors to
  infer a boundary effect.
- LN none/direct2D outputs have their own direct comparison. Unmasked LN is
  only allowed for unpadded columns; target residual width1024 satisfies this.
- Observed JAX mean/variance/inverse statistics include shape/dtype and bounded
  values. Compare instrumented output against uninstrumented JAX. Extra outputs
  can change fusion; these are observed JAX graphs, not Pallas internal nodes
  or definitive evidence about original machine rounding.
- Pin JAX/jaxlib0.10.2 and libtpu0.0.42.1. Record actual device kind and runtime;
  do not infer hardware from the requested Kaggle accelerator label. One
  active device, one Kaggle TPU session at a time.
- Compile all members of each comparison before timing. Five warmups,
  twelve interleaved forward/reverse rounds, resident arguments, synchronous
  completion. Isolated groups and full graphs are separate comparisons.
- Eight retained calls to the same executable, five queue batches, are an
  amortization diagnostic. They are not a real128-chunk scan.
- Profile the two legal16K baselines and all successfully compiled legal16K
  configurations for three calls each, even rejected numerical candidates.
  Profiles follow timed samples. No profile result is asserted before download.
- Save StableHLO before compilation, compiled HLO, first-execution and compile
  times, static memory analysis, strict incremental JSON, timing samples/order,
  direct witnesses and failures. Static byte counts are not hardware counters.
- A failed paired group yields incomparable singleton diagnostics only. No
  failed compilation is timed. No executed full configuration is a fatal
  benchmark error; a failed numerical candidate is not a benchmark crash.
- Existing output JSON is never overwritten. Full runner log and Kaggle log
  remain separate artifacts under the versioned result directory.

## Local verification and launch contract

Fresh pre-change suite:252 passed. New behavior was tested through failing
tests before implementation, including unsupported gather, strict JSON nodes,
provenance mismatch and all-full-cases-failed handling. Post-change suite:
**281 passed in102.81s** on local Python3.11/JAX0.10.1 CPU. Tiny end-to-end tests
execute both corpora, compile/save HLO, preserve failures, reject overwrite and
actually advance a test-only exact candidate from batch2 to4. This is protocol
verification, not target performance evidence. CLI help passes with `src` on
`PYTHONPATH`, as in the launcher.

Source is published first to `TryDotAtwo/TPUBeamSearch`, then the private
`trydotatwo/tpu-execution-boundary-ab` launcher is pinned to that full SHA.
Read all relevant existing TPU statuses before submitting. A denied or unknown
status is not a free slot. Record the actual submission/version/status in a
separate launch record. Terminal analysis must retain failed cases, apply the
frozen gates, publish scoped artifacts and retire its monitor.
