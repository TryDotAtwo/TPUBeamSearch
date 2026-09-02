# Artgor LayerNorm arithmetic attribution v4

Private Kaggle kernel
[`trydotatwo/tpu-artgor-layernorm-arithmetic-attribution`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-layernorm-arithmetic-attribution),
version 4, completed on 2026-09-02.  Source commit:
`fee05807dc82a836bf3c8c17aba03131033b86c9`.

## Result

The run is a valid **rejection**, not a runtime error.  It used eight TPU v5
lite devices, JAX/JAXLIB 0.10.2 and libtpu 0.0.42.1.  Checkpoint, model and
puzzle hashes match the preceding all-Pallas diagnostic.  Six production-size
cases were evaluated at 256 rows/device (2,048 rows globally).

The causal boundary is now narrower:

- The baseline Pallas `hlo_mixed` mean is bitwise exact against the decomposed
  same-call JAX expression on all six cases.
- The first Pallas mismatch is `centered = FP32(values) - FP32(mean)`, on
  2,087,923--2,096,937 of 2,097,152 elements.  The errors are small:
  max abs 0.000244140625--0.0009765625 and mean abs
  5.67e-5--3.51e-4.
- The later BF16 variance happens to be exact on all six cases despite the
  centered FP32 drift.  This is rounding convergence, not evidence that the
  centered values match.
- Baseline final ReLU differs in 445,631--550,725 BF16 elements, max abs
  0.015625.
- Changing variance, epsilon, inverse-standard-deviation or affine precision
  cannot repair the earlier centered mismatch.  The FP32-mean arm diverges one
  boundary earlier, at mean, and is rejected.

## JAX controls

JAX itself is boundary-sensitive for this expression:

- original monolithic LayerNorm+ReLU versus same-call decomposed JAX differs in
  424,968--520,663 BF16 outputs;
- original monolithic versus separately materialized JAX differs in
  132,523--175,840 BF16 outputs;
- same-call versus materialized JAX first diverges at centered subtraction.

These controls do **not** replace the unchanged monolithic Artgor model as the
correctness oracle.  They show that matching source-level algebra is
insufficient: lowering and materialization boundaries change TPU arithmetic.

## Decision and next experiment

No candidate is promoted, timed or fused.  The next minimal experiment must
hold one device-resident BF16 Dense output and one exact BF16 mean fixed, then
compare a standalone subtraction-only Pallas custom call against:

1. same-call JAX subtraction;
2. separately materialized JAX casts and subtraction;
3. Pallas interpret mode;
4. one fused centered-to-variance control.

The next result must add per-checkpoint SHA-256, RMSE and StableHLO/module
identity; v4 recorded mismatch counts and max/mean error but omitted those
fields.  If standalone Pallas matches materialized JAX but not monolithic JAX,
the remaining issue is the oracle's lowering boundary.  If real Pallas differs
from both interpret mode and materialized JAX on the fixed operands, the issue
is in Mosaic subtraction/load precision or lowering.

The safe machine-readable result is
[`artgor_layernorm_attribution.json`](artgor_layernorm_attribution/artgor_layernorm_attribution.json).
Raw private logs remain local and are not published.
