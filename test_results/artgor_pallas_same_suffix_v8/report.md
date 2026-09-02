# v8: pre-rounding mean provenance

Source `bcb8d95e15b969438146f573f7dc005bac39d938`, JAX/jaxlib 0.10.2,
libtpu 0.0.42.1, eight TPU v5 lite devices, 256 states/device. Checkpoint,
model-source and both input hashes match v7. Kernel and JSON completed.

Counts below are directly paired BF16 mismatches. Hidden population is
2,097,152; Q population is 61,440. No timing or speed promotion is made.

| Corpus | Dot rounding | Mean input | Rounded Dense | Prefix | Isolated LN control | Same-suffix Q | Original Q |
|---|---|---|---:|---:|---:|---:|---:|
| legal42 | late | FP32 | 0 | 0 | 21982 | 0 | 44125 |
| legal42 | late | BF16 | 0 | 21982 | 0 | 9624 | 44457 |
| legal42 | before bias | FP32 | 526926 | 365879 | 134455 | 47201 | 47411 |
| legal42 | before bias | BF16 | 526926 | 444909 | 0 | 47780 | 47614 |
| stress43 | late | FP32 | 0 | 15 | 24876 | 17 | 53854 |
| stress43 | late | BF16 | 0 | 24861 | 0 | 27957 | 53905 |
| stress43 | before bias | FP32 | 542737 | 259289 | 38258 | 53972 | 54214 |
| stress43 | before bias | BF16 | 542737 | 272230 | 0 | 53716 | 54262 |

## Established scope

Using pre-rounding FP32 Dense+bias for the mean, while retaining BF16 values
for centering, reproduces the legal input prefix exactly (including SHA).
It reduces stress-prefix disagreement from 24861 elements to 15, but does not
pass the exactness gate. Both BF16-mean controls reproduce their isolated JAX
LN exactly, so this is not a generic failure of the new raw-input LN wrapper.
Before-bias dot rounding fails even the rounded Dense gate and is rejected.

The first remaining stress mismatch is [1085,42]: reference 0.58984375 versus
candidate 0.59375. This is evidence of a remaining discrepancy, not evidence
that all 15 errors share one cause. JAX input-prefix compiled HLO reduces the
FP32 bias-add and separately returns its BF16 conversion. The experiment
supports that provenance hypothesis; HLO alone does not establish the precise
physical accumulation or reduction order.

The original-Q column still fails badly, including the exact legal prefix.
The separately compiled JAX suffix is itself different from the monolithic
oracle. Consequently prefix exactness cannot be promoted to full-model
exactness. The unchanged full-Pallas composition still has 45926/54013 Q
mismatches, since the raw-mean probe is not wired into that engine.

## Next controlled experiment (not yet launched)

Keep the original oracle and all controls. Cross input Dense BK128/BK1024 with
FP32 raw outputs; compare those outputs directly before BF16 conversion.
On exactly the same raw buffers, compare JAX and Pallas row sums, rounded
means, centered values, variance and final LN output. Retain the compiled
prefix comparison: exporting intermediates can itself change JAX compilation.
Save all mismatch coordinates and affected raw rows, not only first mismatch.
Distinguish Dense accumulation differences from reduction-tree differences;
do not patch tolerances or extrapolate one corpus to the six-corpus gate.
Only after both prefix cases are exact should this approach extend to residual
composition. Production defaults, BN and beam search remain unchanged.
