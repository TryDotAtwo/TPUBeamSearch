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
