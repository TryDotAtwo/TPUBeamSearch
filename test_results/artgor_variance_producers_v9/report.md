# V9: standalone JAX producers reproduce Pallas, not the large prefix

Completed source88a97de24c54b7b42d61dd9522927e772e3db5b5, launcher8d0b60c.
All12 cases completed; all comparisons finite. Input/checkpoint/model/puzzle
hashes match v8. Fixed-v4 controls and mean/invstd hashes pass at both shapes.
The extra variance capture remains invalid and is not the oracle.

Eight TPU v5 lite, full131072 legal42 states,16K/device and chunk256/device.
Each producer consumes actual saved v4 Dense/mean, without recomputing mean.

| Producer | invstd differences16K | prefix differences16K | invstd/prefix differences chunk256 |
|---|---:|---:|---:|
| FP32 fused | 2048 | 1275 | 0 / 0 |
| FP32 squares, materialized, FP32 reduction | 2048 | 1275 | 0 / 0 |
| FP32 squares, materialized, original reduction | 2048 | 1275 | 0 / 0 |
| Original-expression fused | 2048 | 1275 | 0 / 0 |
| Original BF16 squares, materialized, FP32 reduction | 990208 | 606952 | 988160 / 605677 |
| Original BF16 squares, materialized, original reduction | 990208 | 606952 | 988160 / 605677 |

The first four match native Pallas invstd exactly. Materializing original BF16
squares differs from its fused producer on988160 broadcast elements (965rows),
at both shapes. Switching the subsequent reduction between FP32/original does
not alter that result. This is evidence for a materialization/rounding effect,
not a fix for the original two-row discrepancy.

Together with v8, standalone JAX FP32 centered-square/reduction and rsqrt
reproduce Pallas, while the unchanged large prefix still differs. The next
diagnosis must preserve producer layout/boundaries, not assume that moving an
expression back to JAX recovers original numerics.

## Limits and next controlled comparison

Initial hidden-FP32-mean speculation is NOT established: validated v4 compiled
HLO presents the variance reduction's mean input as a BF16 scalar parameter,
and the Dense as a BF16 matrix. Its matrix layout is batch-minor `{0,1}`.
Compare the actual standalone producer layout/reduction lowering with this
reference before selecting a new implementation. HLO syntax alone still
does not prove every physical rounding operation.

Concrete lowering difference: `producer_fp32_fused_16384.compiled.txt` takes
packed `[16384,2,1024]` in layout `{2,1,0}` and reduces axes `{1,2}` after
slicing. The original v4 reduction takes separate BF16 Dense `[16384,1024]`
in `{0,1}` and a BF16 mean vector, reducing one feature axis. Equal mathematical
inputs therefore have not established equal physical reduction geometry.
Next use separate matrix and scalar-mean arguments, preserve rank and inspect
compiled layout before concluding that producer arithmetic itself differs.

Square hashes differ between shapes, but the supplied means already differ
between shapes on one row. Therefore unequal square hashes are not by
themselves evidence of a shape-dependent square operator. A next experiment
must hold BOTH Dense and mean identical across compilation shapes and examine
layout-matched/transposed reduction, with native/original output controls.

No all-Pallas/full-model correctness or speed claim. Borrowed JAX statistics
remain diagnostic. JSON and scalar results are exhaustive; bounded NPZ are
examples only. Large original JAX stays the oracle.
