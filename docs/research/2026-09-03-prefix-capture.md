# Controlled BF16 prefix capture

Inference-only follow-up to `test_results/artgor_prefix_shape_v2/report.md`.
The unchanged large-shape JAX oracle remains mandatory. No speed promotion.

Use the identical legal42 corpus, 131072 states partitioned over eight TPU.
Enforce its previous input SHA and reproduce both untouched large output SHAs.
Run at 16384 and 256 rows/device, preserving device-major chunk reconstruction.

Capture BF16 Dense, broadcast BF16 mean, and output together for JAX and Pallas.
Compare captured output to each untouched executable. Extra outputs can change
fusion: a failing capture control forbids attribution to the untouched oracle.
Record Dense/mean/output shape comparisons, not just output differences.

Cross JAX/Pallas materialized Dense and mean through the same Pallas remainder.
The Pallas/Pallas substitution must reproduce the untouched Pallas prefix.
The other three substitutions remain diagnostic if either capture or zero-change
control fails. In particular, matching a substituted output alone proves nothing
about the arithmetic of an invalid captured reference.

Save finite, bitwise/numerical/signed-zero counts, complete SHA-256, affected
state/output rows in NPZ, and compiled HLO/StableHLO for captures, controls and
the shared remainder. CPU tests validate the capture's dtype/slots only; TPU
execution decides whether compiler instrumentation is non-perturbing.

## Invstd follow-up

`--include-invstd` adds slot 3, a genuinely BF16 broadcast inverse standard
deviation. Slots 0/1/2 and their output controls remain unchanged. The JAX
capture computes invstd using the original source arithmetic, not a manually
rewritten FP32 approximation; its output still must match the untouched call.
The Pallas capture returns the existing output plus an independently computed
BF16 invstd buffer. A native external-affine zero-control checks whether this
additional materialization reproduces the existing Pallas output.

For causal substitution, fix BOTH Dense and mean to captured JAX values.
Recompute Pallas invstd from those fixed inputs (do not substitute an invstd
computed with a different mean). Pass either Pallas or JAX invstd through the
same Pallas affine/ReLU kernel. Compare the Pallas split path against the
existing external-mean LN kernel on those exact inputs. Any failed capture,
native-affine or fixed-mean split control invalidates attribution.

Record all original comparisons plus captured/fixed-mean invstd, two affine
outputs, and both new zero-controls. Export HLO for auxiliary executables as
well. Unit/interpreter tests cover the BF16 capture slot and externally supplied
invstd affecting output; they are not TPU proof. No variance-trace widening is
introduced in this step, and no production defaults change.

## Variance follow-up

`--include-variance` retains the untouched reference and v4 capture as controls,
adding actual BF16 variance in slot4 of the new JAX capture. Both output and
invstd must equal the v4 capture, and the v4 invstd SHA must reproduce the
published run. No entire-trace widening is used.

On fixed JAX Dense/mean, Pallas returns FP32 variance and BF16 invstd as two
separately typed outputs from one kernel. Native invstd must reconstruct v4's
producer. Five reduction orders use identical centered-square inputs. Each
variance is replayed from a real FP32 or BF16 buffer through the same Pallas
epsilon/rsqrt kernel; the FP32 replay has its own native reconstruction check.
The captured JAX BF16 variance is also replayed through JAX and Pallas. A failed
replay control is a boundary effect, not evidence for reduction causality.

Retain all previous controls, both complete-corpus shapes, HLO, mismatch rows
and scalar NPZ files with full FP32 variance and BF16 invstd/variance bits.
The pair-output host collector preserves device-major order and distinct dtypes.
This is attribution only: no production kernel or default is promoted.

## Fixed-v4-input follow-up

`--use-v4-inputs` explicitly selects the unchanged four-slot v4 capture for
Dense, mean, invstd and output. The five-slot variance capture stays separate;
its failed controls cannot supply the inference mean. Every shared field is
compared. Original v4 mean/invstd SHA and exact JAX-statistics/Pallas-affine
reconstruction remain mandatory. `diagnostic_variance_valid` is distinct from
the validity of these fixed-input comparisons; BF16 variance comparisons and
JAX-variance replays remain diagnostic-only when that flag is false.

Each reduction candidate now also produces a full prefix output via the same
Pallas affine with fixed v4 Dense/mean. All tensor counts/SHA remain exhaustive,
but NPZ mismatch examples are limited to eight rows (`examples_only=true`).
Complete scalar variance/invstd bits are retained for every corpus row.

## Same-buffer rsqrt consumer follow-up

`benchmarks.artgor_rsqrt_consumers.consume_variance` supplies paired JAX/Pallas
consumers for explicit FP32 arithmetic and a separate BF16 source-expression
control. FP32 mode uses the original epsilon quantized to BF16 then widened,
so differences do not silently include a new epsilon constant. Scalar 1D and
broadcast 2D storage are separate executions; do not broadcast inside the
scalar kernel and call it a scalar-layout experiment. Interpreter tests establish
only shape/dtype/epsilon routing and distinguish the arithmetic modes.

Integration must use the native FP32 variance returned by the validated v6
pair producer, preserve it as a real input argument and retain v4 output/hash
controls. Export each lowered consumer and compare JAX/Pallas directly, against
validated invstd and through the same Pallas affine. Reconstruct scalar results
to broadcast on the host after execution for prefix comparisons. Retain full
scalar bits and shape-correct sharding (`P('core')` for rank one). Test the
collector's device-major ordering before launching. No standalone consumer
result licenses an all-Pallas or speed claim.

`--compare-consumers` now enables this matrix in the capture runner, implying
`--use-v4-inputs`. It evaluates eight consumers per shape on the native pair's
same FP32 variance: JAX/Pallas x scalar/broadcast x FP32/BF16-expression.
Each pair records direct equality, equality to native Pallas and validated JAX
invstd, and prefix reconstruction. Scalar input/output bits are saved for all
rows; lowered consumer and affine HLO are saved separately. Existing five-order
comparisons remain intact. A compilation failure terminates the diagnostic
with partial JSON rather than being interpreted as a numerical/timing result.

## Fixed-input variance producer A/B

`--compare-producers` implies validated v4 inputs. At both full-corpus shapes,
JAX centered squares use either the original BF16 expression or explicit FP32.
Each is tested fused through mean/epsilon/rsqrt, and separately materialized
through original or FP32 reductions (six variants/shape). Fixed mean is never
recomputed. Scalar invstd is compared against v4, native Pallas and the matching
fused producer; shared Pallas affine reconstructs the prefix against untouched
JAX. Square buffers have exhaustive hashes/dtypes/finite and shape records;
all final scalar bits and HLO are retained. No new variance output is attached
to the v4 oracle, so prior instrumentation failures remain isolated.

Materialized-original vs fused-original is a boundary comparison, not merely
a reduction algorithm comparison. Explicit FP32 reduction of BF16 squares is
not equivalent to producing squares in FP32. The unchanged large oracle and
its controls remain mandatory; no winner or speed claim before TPU evidence.
