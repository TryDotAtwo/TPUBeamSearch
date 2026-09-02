# Monolithic LayerNorm arithmetic match v1

Private Kaggle kernel `trydotatwo/tpu-artgor-layernorm-monolithic-match` v1
completed from source `a50f6490abc6e65428d73a86ab9ae1122ace28d3`
on eight TPUv5lite devices.

## Result

The real one-kernel Pallas `fp32_variance` arm is bitwise identical to the
unchanged monolithic Artgor JAX LayerNorm on all six frozen B256/device
corpora: zero BF16 mismatches and identical SHA-256 in every case.

Every other Pallas arm fails. The baseline materialized-style arm differs by
132,523--175,840 elements. The explicit JAX one-factor arms also all fail,
including `fp32_variance`; this is expected evidence that placing the same
source expression behind a different JIT/materialization boundary changes its
lowering. It does not invalidate the directly measured Pallas equality.

## Exact arithmetic contract

Inside one Pallas LayerNorm call:

- BF16 input is converted to FP32;
- mean is reduced in FP32 and rounded to BF16;
- centered values remain FP32;
- variance remains FP32 (the decisive change);
- inverse standard deviation is rounded to BF16;
- affine is evaluated in FP32 and rounded once to BF16;
- optional residual add and ReLU remain inside the call.

This contract is now promoted into the full 44-stage all-Pallas diagnostic.
Timing remains blocked until every model boundary and final Q are hash exact.

