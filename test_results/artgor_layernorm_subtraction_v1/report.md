# Fixed-operand LayerNorm subtraction v1

Private Kaggle kernel
[`trydotatwo/tpu-artgor-layernorm-subtraction`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-layernorm-subtraction),
version 1, completed on 2026-09-02 from source
`a5b7690fd0e3b24a98e26fe2c134b93308107762`.

## Result

The centered-subtraction contract is exact once its BF16 operands are explicit
dispatch inputs.  The run used eight TPU v5 lite devices, JAX/JAXLIB 0.10.2,
libtpu 0.0.42.1, 256 rows/device and all six frozen legal/stress corpora.
Checkpoint, model and puzzle hashes match the preceding diagnostics.

For every corpus and every 2,097,152-element centered tensor:

- same-call JAX subtraction equals separately materialized JAX casts and
  subtraction, bit for bit;
- standalone real-TPU Pallas equals materialized JAX, bit for bit;
- Pallas interpret mode equals both real Pallas and materialized JAX;
- a Pallas kernel that consumes the same explicit values/mean and immediately
  reduces centered values to BF16 variance equals materialized JAX variance.

All comparisons have zero mismatches, max/mean/RMSE zero and matching SHA-256.
The recorded StableHLO identities are distinct.  Real standalone Pallas and
fused variance each contain one `tpu_custom_call`; interpret mode contains none.

## Attribution

The v4 LayerNorm mismatch is **not** an inherent precision limitation of Pallas
FP32 subtraction.  It arises when mean production and centered subtraction are
kept inside the larger Pallas LayerNorm kernel.  Making the already exact BF16
mean an explicit materialized input removes the drift completely.  The same
explicit boundary remains exact when centered values are consumed immediately
by variance, so the variance reduction itself is not the source.

The correctness-first implementation should therefore split each LayerNorm at
the mean boundary: one Pallas mean dispatch followed by a Pallas remainder
dispatch.  This temporarily increases dispatch count beyond the planned
44-boundary baseline.  Only after full-model bitwise equality should an
in-kernel VMEM materialization/barrier be investigated to recover one dispatch
per LayerNorm.

The safe raw result is
[`artgor_layernorm_subtraction.json`](artgor_layernorm_subtraction/artgor_layernorm_subtraction.json).
Private logs are retained locally and are not published.
