# Residual skip-rounding A/B v6

Source `53f42155406bda72aee72ea8f10cf5927da68f2d`; eight TPU v5 lite devices,
256 states/device, legal42/stress43. This is operator validation, not full-model
or 16K/32K promotion.

| LN2+skip arithmetic | Hidden mismatches legal/stress | Same-suffix Q mismatches |
|---|---:|---:|
| FP32 variance, early skip rounding | 249092 / 192197 | 44330 / 53773 |
| BF16 variance, early skip rounding | 326448 / 236737 | 44691 / 53740 |
| BF16 variance, late skip rounding | 104256 / 58954 | 4768 / 4204 |
| FP32 variance, late skip rounding | **0 / 0** | **0 / 0** |

The late FP32-variance candidate matches the isolated JAX operator bitwise on
both corpora, with identical hashes and exact zero-replacement checks. Dense1,
Dense2 and LN1 controls remain exact in their previously selected configurations.

The visible JAX compiled HLO contains a BF16 variance conversion, but the measured
winner retains FP32 variance. Therefore the HLO text alone does not establish
the effective physical rounding at that point; claiming that BF16 variance was
the correct fix would contradict the A/B. Late skip rounding is supported by
the factorial comparison and the half-ULP witness.

Next: select FP32 variance for normal LN and FP32 variance + late rounding for
skip LN via an optional config field. Keep defaults unchanged. Re-evaluate
isolated blocks and full all-Pallas composition against unchanged JAX at B256,
and capture both input-prefix and full-model HLO. A remaining prefix-scope
discrepancy is expected to require separate investigation; no speed claim yet.
