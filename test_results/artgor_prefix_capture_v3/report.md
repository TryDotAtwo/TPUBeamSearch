# Prefix capture v3: exact Dense, one mean row, two remaining LN rows

Kaggle v3 completed with source `489b69ba09583c7418b8b29ecd1f70fb1c04520e`
and launcher `e1def40`. Eight TPU v5 lite, JAX/jaxlib 0.10.2,
libtpu 0.0.42.1. Input, checkpoint, model-source hashes and runtime versions
match shape v2; both untouched large-output hashes reproduce the prior gate.
Puzzle SHA is recorded in JSON (not independently matched to shape v2, which
did not record that field). The inherited inventory's `active_device_count=1`
is not a measured shard count: this runner explicitly uses an eight-device
Mesh and records all eight devices. No timing/utilization claim is made.

All comparisons are finite, with zero signed-zero discrepancies. Both JAX
and Pallas capture-output controls and the Pallas zero-replacement control
are bitwise exact at both batch shapes. Therefore the observed capture outputs
are non-perturbing for this corpus.

| Comparison | 16K/device mismatches | chunk256/device mismatches |
|---|---:|---:|
| Captured BF16 Dense, JAX vs Pallas | 0 | 0 |
| Captured broadcast mean | 1024 (one row) | 0 |
| Untouched prefix output | 1329 (three rows) | 0 |
| Pallas Dense + Pallas mean, shared remainder | 1329 | 0 |
| JAX Dense + Pallas mean, shared remainder | 1329 | 0 |
| Pallas Dense + JAX mean, shared remainder | 1275 | 0 |
| JAX Dense + JAX mean, shared remainder | 1275 | 0 |

The mean discrepancy is one scalar broadcast to 1024 columns on global row
54401: JAX `0.2099609375`, Pallas `0.208984375`. Substituting JAX's mean removes
all 54 output differences on that row. Remaining rows are 760 (661 elements)
and 28870 (614): 1275 total. Swapping BF16 Dense has no effect, consistent with
its identical complete SHA. NPZ row coordinates independently confirm this.

JAX large/chunk Dense is exact; its mean differs on the same one row and output
on the same three rows. Pallas Dense, mean and output are each shape-invariant.
At chunk256 every substitution is exact over the entire same 131072-state
corpus, not merely a smaller independent sample.

## Attribution limits and HLO

This isolates a mean contribution plus a remaining difference in the LN
remainder. It does not yet identify variance reduction, reciprocal square root,
or affine rounding as the remaining cause. Equal BF16 Dense buffers do not
prove equal pre-round FP32 accumulators feeding the fused mean.

Compiled capture HLO at 16K (`jax_capture_16384.compiled.txt`, lines 129-144)
keeps Dense/bias and the mean reduction together, returning a BF16 Dense buffer
and FP32 reduced sum. Lines 163-171 describe centered FP32 square/reduction;
lines 174-186 describe variance scaling, BF16 conversion, epsilon and rsqrt.
The 256 HLO has the analogous sequence with different layouts. These are
compiler observations, not proof of physical rounding or reduction order.
Prior experiments specifically showed why visible conversion syntax alone
is insufficient.

## Next controlled experiment

Capture the actual BF16 inverse standard deviation alongside Dense, mean and
unchanged output, at both shapes on this exact corpus. Repeat capture-output
controls before attribution. Swap JAX/Pallas inverse standard deviations into
one Pallas affine/ReLU remainder with fixed JAX Dense/mean; retain a native
zero-control. If this removes the remaining two rows, localize further with
variance/reduction captures; if not, test affine rounding. Any added variance
trace needs its own output control and must not be mistaken for an untouched
intermediate after a failed control. Do not shrink/reorder the corpus to just
the affected rows, which would change the compilation shape under investigation.

The expert consultation timed out; no expert conclusion is claimed. Raw JSON,
NPZ, HLO and both logs are included. This is prefix-only evidence, not full-model
correctness or a new speed result. The large JAX oracle and production defaults
are unchanged.
