# V8: matched FP32 rsqrt consumers agree; prefix discrepancy is upstream

Completed source `676a3764c78a01ff1b352e98d5062b978df5b2c6`, launcher `ac5f84e`.
All16 consumer cases compiled and completed on eight TPU v5 lite devices.
JAX/jaxlib0.10.2, libtpu0.0.42.1; input/checkpoint/model-source/puzzle hashes
match v6. All comparisons are finite. Both fixed-v4 control sets are valid;
mean/invstd SHA reproduce. Extra variance capture remains invalid and separate.

## Same materialized input, direct equality

All consumers receive the same native Pallas FP32 variance as a runtime input.
Scalar1D and broadcast2D are separate TPU calls; scalar output broadcasting
occurs only after execution on the host. The corpus is all131072 legal42 states,
partitioned across eight devices at16K/device or reconstructed chunk256/device.

| Consumer group (both layouts) | invstd mismatches vs JAX16K | prefix mismatches vs JAX16K | invstd/prefix vs chunk256 JAX |
|---|---:|---:|---:|
| Explicit FP32 JAX | 2048 | 1275 | 0 / 0 |
| Explicit FP32 Pallas | 2048 | 1275 | 0 / 0 |
| BF16 source-expression JAX | 2048 | 1275 | 0 / 0 |
| BF16 source-expression Pallas | 16686080 | 10233096 | 16684032 / 10231815 |

Explicit FP32 JAX and Pallas have ZERO direct mismatches, and both reconstruct
the native Pallas invstd exactly, at both shapes and layouts. Thus changing
only the FP32 epsilon+rsqrt consumer to JAX does not fix the original prefix.
Within this tested contract, the unresolved discrepancy is upstream of this
consumer, in the variance producer or its boundary. This is not a proof that
all rsqrt lowerings on all inputs agree.

BF16-expression JAX also reconstructs native FP32 invstd, whereas BF16-expression
Pallas differs from it on16684032 broadcast elements =16293 distinct rows.
This source-expression discrepancy is separate from the original two-row
invstd discrepancy. A BF16 cast in source cannot be treated as an equivalent
physical rounding boundary across the two compilers. Neither branch should
replace the validated reference without a full output control.

Compiled JAX scalar HLO has an FP32 add/rsqrt in both variants. The BF16-source
variant additionally displays an input BF16 roundtrip plus float-type-correction
metadata, yet its measured output equals the explicit-FP32 variant. This
reinforces why printed conversion syntax is not proof of physical rounding.
Full scalar NPZ confirm identical input variance and outputs across scalar JAX
and broadcast Pallas FP32; reference differences are rows760/28870 only.

## Scope and next experiment

The reference still uses its original large-shape prefix. Borrowed JAX
statistics do not constitute an all-Pallas engine. No full-model correctness,
latency, throughput or speedup claim is made. Matrix and scalar FP32 consumers
are equally exact on this corpus; this is not a performance comparison.

Next isolate the upstream producer: compare JAX-only centered-square/reduction
from the same validated Dense/mean buffers, with output/invstd reconstruction
controls and original-expression vs explicit-FP32 variants. Returning variance
must not silently perturb the reference. If a new capture fails, retain it as
diagnostic only. Compare materialized centered-square inputs separately from
fused reduction before attributing the difference to summation order. Preserve
both shapes, all corpus rows and the original oracle; do not tune two rows.

JSON is exhaustive; mismatch NPZ are bounded examples, scalar NPZ cover every
row. V7's BF16 rank-one block rejection is resolved on TPU in this run.
