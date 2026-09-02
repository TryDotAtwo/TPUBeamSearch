# LayerNorm boundary replay v1

Private Kaggle kernel `trydotatwo/tpu-artgor-layernorm-boundary-replay` v1
completed from source `5ee2b43719addd5c2e205f61039e2f6ddd07274c`
on eight TPUv5lite devices. The lowered replay contains exactly five
`tpu_custom_call` operations, matching the five explicit Pallas boundaries.

## Causal result

Modular Pallas is bitwise identical to separately materialized JAX at every
observed boundary on all six frozen B256/device corpora:

- BF16 mean;
- FP32 centered tensor;
- BF16 variance;
- BF16 inverse standard deviation;
- BF16 affine plus ReLU.

Every boundary has zero mismatches, zero error and the same SHA-256. Therefore
the isolated Pallas primitives and their composition are correct for the
explicitly materialized arithmetic contract.

The unchanged monolithic Artgor JAX LayerNorm is different. Its final BF16
output differs from both modular Pallas and materialized JAX by 132,523--175,840
elements, max abs 0.015625--0.03125 and RMSE 0.000663--0.000728. Modular Pallas
and materialized JAX have the same final hash in every corpus.

## Consequence

The remaining correctness problem is not a defective Pallas primitive. The
monolithic JAX lowering evaluates an unmaterialized mixed-precision expression
with different effective rounding boundaries. The next experiment must match
that lowering directly by testing one-factor arithmetic variants against the
monolithic output. Performance optimization remains blocked until one variant
is hash exact.

