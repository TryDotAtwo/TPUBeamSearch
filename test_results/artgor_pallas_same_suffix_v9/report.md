# v9: raw accumulation and instrumentation controls

Source `c1ee40a041bfbc21c5cd15937fd97568da82d9ae`; diagnostic batch 256/device
on eight TPU v5 lite. These are numerical tests, not performance measurements.

| Corpus | BK | Rounded Dense mismatches | Prefix mismatches | Same-suffix Q mismatches | Pallas trace vs uninstrumented |
|---|---:|---:|---:|---:|---:|
| legal42 | 128 | 0 | 0 | 0 | 0 |
| legal42 | 1024 | 120 | 28 | 415 | 0 |
| stress43 | 128 | 0 | 15 | 17 | 0 |
| stress43 | 1024 | 174 | 21 | 450 | 0 |

Raw FP32 Dense differs between BK128 and BK1024 in 1,420,314 legal and
1,487,480 stress elements out of 2,097,152. Thus larger BK changes accumulation
and fails even the BF16 Dense boundary. Reject BK1024 for this input layer;
this does not contradict isolated residual Dense results with other dimensions.

The Pallas multi-output trace retains exactly the uninstrumented output in all
four cases. However the JAX trace is NOT an exact arithmetic reference: its
field named `mean_bf16`, widened back to FP32 before returning, contains e.g.
0.3417632281780243 rather than the BF16-representable 0.341796875. Similar
effects occur in centered values and inverse standard deviation. These actual
outputs show that source-level BF16 round trips are insufficient to enforce
rounding in this compiled diagnostic. Consequently its large later-stage
mismatch counts must not be attributed to the model or a Pallas LN defect.

The same-buffer FP32 sum also differs, establishing a reduction implementation
difference, but not proving it causes the remaining 15 stress-prefix errors.
Both accumulation and reduction can differ while rounded boundaries remain
equal. The next test must determine whether the row mean actually crosses a
BF16 rounding boundary on the affected rows.

All 15 BK128 stress mismatches belong to row 1085; the saved raw/reference/
candidate rows are finite. CPU FP64 summation of that saved FP32 row gives
mean 0.028259291175345425, only 1.3831591675e-8 above the BF16 midpoint
0.02825927734375 (neighbors 0.0281982421875 and 0.0283203125). This makes mean
rounding a concrete next hypothesis, not proof of the TPU reference mean:
the reference's unrounded Dense and physical sum are not captured here.

The trace HLO still contains BF16 converts despite the non-BF16 values observed
after widening. This reinforces the existing rule to verify effective rounding
from device results rather than dtype annotations alone. Runtime, checkpoint,
model-source and both input hashes match v8.

## Next controlled test

Retain BK128. Return JAX row sums in FP32 and rounded means in genuine BF16
buffers; materialize these separately before feeding a Pallas remainder with
an externally supplied mean. Compare native Pallas mean against JAX mean and
each remainder against the existing uninstrumented output and JAX prefix.
Keep a zero-change external-mean control. Do not infer physical monolithic
intermediates from exported JAX traces. Preserve original full Q as the final
gate; no residual expansion or default change until the input prefix passes.

NPZ files preserve all prefix mismatch coordinates and affected raw rows.
Full original-Q equality remains unachieved; no new speedup is established.
