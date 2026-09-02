# Residual0 operator arithmetic A/B v5

Eight TPU v5 lite devices, B256/device; legal42/stress43. Source
`6265c83956ef038e1010e88a695ee257696606d2`. Diagnostic only.

- Dense1 and Dense2: BK128 and BK1024 both exact on identical inputs.
- LN1+ReLU: FP32-variance candidate exact. HLO-mixed BF16 variance and legacy
  arithmetic are inexact.
- LN2+skip+ReLU: all three candidates are inexact. Current FP32 variance has
  249092/192197 hidden mismatches and 44330/53773 same-suffix Q mismatches.
- Prefix-versus-isolated-reference differences and zero-replacement results
  remain separate; no complete-model equivalence is claimed.

## Structural cause and falsifiable follow-up

The JAX `layernorm2_skip_relu-jax.compiled.txt` contains a BF16 conversion of
the reduced variance before epsilon/rsqrt. Its final fusion retains centered
normalization, affine bias, residual addition and ReLU in FP32, converting to
BF16 only at the output. In contrast, the current Pallas branch converts affine
output to BF16 before skip addition; current FP32-variance also omits the
variance conversion. These are two independently testable arithmetic differences.

v6 retains the old early-round controls and adds late-skip variants with BF16
and FP32 variance. A CPU/Pallas-interpreter half-ULP witness proves that moving
the rounding changes behavior: `1 + 1/256 - 1` becomes zero with early BF16
rounding but remains `1/256` with late rounding. This is not yet TPU equivalence
evidence. Production defaults are unchanged; unchanged full JAX remains gate.
