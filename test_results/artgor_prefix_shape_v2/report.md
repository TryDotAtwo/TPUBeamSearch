# Same-state shape diagnostic v2: JAX changes, Pallas is invariant

Source `5724a2745bc8785ea1e9ccc81922959fff371d02`; one legal42 corpus of
131072 states, partitioned over eight TPU v5 lite. Its input SHA is enforced
against prefix gate v1. Runtime/checkpoint/model-source match; both full-size
output hashes exactly reproduce the earlier failed gate.

| Comparison on identical states | Bitwise mismatches | Numeric mismatches | Signed-zero differences |
|---|---:|---:|---:|
| JAX 16K/device vs JAX chunk256/device | 1329 | 1329 | 0 |
| Pallas 16K/device vs Pallas chunk256/device | 0 | 0 | 0 |
| JAX vs Pallas, both 16K/device | 1329 | 1329 | 0 |
| JAX vs Pallas, both chunk256/device | 0 | 0 | 0 |

All outputs are finite. The 1329 mismatches occupy only three global rows:
760 (661 elements), 28870 (614), and 54401 (54). Both Pallas runs and chunked
JAX have identical complete output SHA. Thus the earlier failed case is not
explained by larger corpus sampling or Pallas composition drift: on precisely
these states the JAX reference changes with compilation shape, while Pallas
reproduces the small-shape JAX result even at large batch.

This does NOT make JAX incorrect and does NOT authorize replacing the oracle.
The requested large-shape JAX prefix is still unmatched. The shape-dependent
HLO matrix layout change `{1,0}` -> `{0,1}` remains an observation; whether the
first arithmetic divergence is Dense accumulation or LayerNorm reduction is
not yet established by these final-prefix comparisons.

## Next controlled boundary test

Retain this exact 16K legal42 corpus. Capture genuine BF16 Dense+bias and mean
buffers alongside the JAX prefix output, checking captured-prefix output
against the uninstrumented oracle before trusting any intermediate. Compare
large and chunk256 captured tensors, and Pallas raw Dense rounded to BF16 plus
its mean. If capture control passes, cross JAX/Pallas Dense values and means
in the SAME Pallas LN remainder; keep native zero-change control. If capture
changes outputs, report the confound rather than attributing its intermediates
to the original computation. Save affected rows/bits and HLO.

The diagnostic capture helper was corrected to retain signed-zero differences;
this run establishes that these particular 1329 differences are numerical,
not merely zero signs. No tolerance, production default, BN or beam change;
no full-model correctness or performance claim.
