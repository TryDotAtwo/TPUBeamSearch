# Same-input same-suffix v4

Source `f1a0a1a731ab992845fe4a81f823e9922ae935e0`; eight TPU v5 lite devices,
JAX 0.10.2, B256/device; legal42 and stress43. Diagnostic, not promotion.

| Operator | Same-input hidden mismatches legal/stress | Same-suffix Q mismatches |
|---|---:|---:|
| Embedding | 0 / 0 | 0 / 0 |
| Input Dense | 0 / 0 | 0 / 0 |
| Input LN+ReLU | 0 / 0 | 0 / 0 |
| Residual 0 | 264009 / 206762 | 44372 / 53795 |
| Head | 0 / 0 | 0 / 0 |

All ten residual blocks remain inexact against isolated JAX blocks consuming
the same runtime hidden tensor. All explicit zero-replacement checks are exact.

Input LN itself is exact against isolated JAX LN. Its reference differs from
the JAX input-prefix output by 21982/24861 BF16 elements. Thus isolated operator
exactness does not imply composition exactness against the full compiled model.
The full input_stack is still inexact; no production gate is relaxed.

Next: residual0 operator factorization on reference-generated, identical inputs.
Dense1/Dense2 compare BK128 and BK1024; LN1/LN2+skip compare current FP32-variance,
hlo_mixed and legacy_bf16 arithmetic (10 candidates total). Keep fused-prefix
versus isolated-reference drift separate. Save compiled and StableHLO for each
operator candidate to support arithmetic attribution. No speed claims.

Expert consultation was attempted; Telegram bridge returned a group migration
error. No recommendation was received or used.
