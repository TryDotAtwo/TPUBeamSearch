# FP32-variance all-Pallas diagnostic v6

Private Kaggle kernel `trydotatwo/tpu-artgor-all-pallas-exact-diagnostic` v6
completed from source `989715789d8eff007fb08c143ccedcfeb8121e27`
on eight TPUv5lite devices.

Embedding and BK128 input Dense are bitwise exact. The first mismatch remains
`input.layernorm_relu` on all six frozen B256/device corpora, but it is reduced
to 21,165--25,312 BF16 elements (max abs 0.0078125--0.015625).

The arithmetic itself is not the cause: the standalone one-call
`fp32_variance` Pallas probe is hash exact against the same monolithic JAX
boundary. Comparing its kernel construction with the production kernel found
one remaining structural difference: production always emitted a column
predicate and `where` operations even when logical and physical widths were
both 1024. The standalone exact probe statically omitted that mask.

This is both a numerical-lowering issue and avoidable vector work. The v7 fix
statically removes the predicate/select path for aligned widths while retaining
masking for genuinely padded widths. Timing remains blocked until the full
44-stage gate is exact.

