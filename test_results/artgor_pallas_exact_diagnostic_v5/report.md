# Fully materialized all-Pallas diagnostic v5

Private Kaggle kernel `trydotatwo/tpu-artgor-all-pallas-exact-diagnostic`
completed on eight TPUv5lite devices from source
`c642863f3fee4e1e2ae170a239245a2dae54097b`.

## Result

The candidate is correctness-rejected before timing. Embedding and the BK128
input Dense are bitwise exact on all six frozen legal/stress corpora. The first
mismatch is `input.layernorm_relu` in every case:

| corpus | mismatched BF16 values | max abs | mean abs |
| --- | ---: | ---: | ---: |
| legal seed 42 | 163,288 | 0.03125 | 0.000147123 |
| legal seed 142 | 168,930 | 0.03125 | 0.000153996 |
| legal seed 242 | 194,400 | 0.03125 | 0.000175400 |
| stress seed 43 | 154,177 | 0.015625 | 0.000145626 |
| stress seed 143 | 162,741 | 0.03125 | 0.000154240 |
| stress seed 243 | 162,114 | 0.015625 | 0.000154381 |

## Interpretation

Five explicit Pallas calls per semantic LayerNorm (mean, centered subtraction,
variance, inverse standard deviation, affine/activation) do not reproduce the
unchanged monolithic `jax_model.apply` boundary. The mismatch counts are
identical to the earlier unsplit and split-mean all-Pallas candidates. This is
not evidence that the isolated primitives are wrong: the fixed-operand TPU
probes already showed each primitive exact. It instead narrows the remaining
question to the arithmetic semantics induced by the monolithic JAX lowering
versus an explicitly materialized sequence.

The next test must compare the exact same real Pallas intermediates against
both (a) monolithic JAX and (b) an explicitly materialized JAX replay on the
same Dense tensor. No performance claim or default-engine change is made.

