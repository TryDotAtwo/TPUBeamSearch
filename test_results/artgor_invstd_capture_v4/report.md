# Invstd capture v4: both remaining rows explained by inverse standard deviation

Kaggle v4 completed from source `4ab805bb2bac756212dfd882cb23f7e0ba3a601f`,
launcher `00f8178`. All runtime metadata, input/checkpoint/model-source/puzzle
hashes match v3. Eight TPU v5 lite, JAX/jaxlib 0.10.2, libtpu 0.0.42.1.
Both untouched large output hashes reproduce the earlier gate. The inherited
inventory `active_device_count=1` is not the runner's shard count: the explicit
mesh has eight devices. This is a numerical experiment, not a timing run.

On the same 131072 legal42 states at 16K/device and chunk256/device, every
capture-output, native-remainder, native-affine and fixed-mean split control
passes bitwise. All comparisons are finite; no signed-zero differences.

| Large-shape comparison | Mismatched elements |
|---|---:|
| Untouched JAX vs Pallas prefix | 1329 |
| JAX mean + Pallas remainder (v3 reproduction) | 1275 |
| Captured JAX vs Pallas invstd | 2048 (two broadcast scalars) |
| JAX invstd vs Pallas invstd recomputed with fixed JAX Dense/mean | 2048 |
| Fixed JAX Dense/mean + Pallas invstd + Pallas affine/ReLU | 1275 |
| Fixed JAX Dense/mean + JAX invstd + same Pallas affine/ReLU | **0** |

The last comparison matches the entire untouched JAX prefix SHA:
`9755606bffa3d179337f5741fcd23dce5f0469d6b11ebc56c546f6e25b6cd7f0`.
The two different invstd scalars are:

| Global row | JAX invstd | Pallas invstd | Prior output differences |
|---|---:|---:|---:|
| 760 | 0.126953125 | 0.1279296875 | 661 |
| 28870 | 0.16015625 | 0.1591796875 | 614 |

JAX invstd changes between large and small shapes only on these rows; Pallas
invstd is shape-invariant. At chunk256, all comparisons and substitutions are
exact on the entire same corpus. The separate mean discrepancy on row54401
and its 54 output errors reproduce v3.

## What this establishes

For this corpus, reproducing the JAX mean and BF16 invstd is sufficient: the
existing Pallas affine/ReLU remainder then reproduces the untouched large JAX
prefix exactly. The fixed-mean invstd recomputation and split zero-control
rule out changing the mean or merely adding a dispatch as the explanation for
removing the two remaining rows.

This is NOT yet an all-Pallas exact prefix: the successful diagnostic borrows
JAX statistics. It is not full-model correctness and is not a speed result.
No production/default, BN or beam change is justified by this diagnostic.

## HLO evidence and remaining uncertainty

`jax_capture_16384.compiled.txt` describes variance reduction and the
variance/epsilon/rsqrt fusion (lines174-186). The resulting BF16 `%fusion.1`
feeds both the output-affine fusion and the captured invstd broadcast in the
entry computation. Combined with unchanged-output controls, this supports
the relevance of the captured invstd. HLO syntax alone does not establish
physical rounding or the reduction tree.

The remaining attribution question is upstream of the affine operation:
centered-square reduction, variance rounding, epsilon addition, or rsqrt
rounding/implementation. BF16 Dense equality still does not prove pre-round
FP32 equality in the fused mean producer. Keep that separate one-row mean
problem distinct from the two-row invstd problem.

## Next controlled variance experiment

1. Add an actual BF16 variance output to JAX capture, retaining Dense, mean,
   invstd and prefix output. Require BOTH output and invstd to reproduce v4.
   Capturing variance may itself alter optimization; reject attribution if
   either control fails. Never widen the entire capture to FP32.
2. On fixed JAX Dense/mean, capture Pallas FP32 variance and native BF16
   invstd together, with a zero-control against v4's invstd producer. Compare
   reduction candidates only with identical centered inputs.
3. Replay actual captured variance buffers through separately controlled
   epsilon+rsqrt kernels. Compare BF16 and FP32 variance consumption and exact
   native reconstruction before claiming reduction or rsqrt causality.
4. Preserve the complete corpus, device ordering and both compilation shapes.
   Save affected scalar values/bits, finite/exact/SHA, and HLO. Do not tune to
   only two rows or relax the large-shape oracle.

Raw JSON, 82 output artifacts and the full Kaggle log are retained in this
directory. Source tests passed 472 before launch; those tests are not TPU proof.
