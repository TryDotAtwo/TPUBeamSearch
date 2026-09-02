# Fixed-variance LayerNorm invstd and affine v1

Private Kaggle kernel
[`trydotatwo/tpu-artgor-layernorm-invstd`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-layernorm-invstd),
version 1, completed on 2026-09-02 from source
`df562624015a2b27b722e14915138d7345c0764b`.

The run used eight TPU v5 lite devices, 256 rows/device and all six frozen
legal/stress corpora.  With exact BF16 variance and explicit centered/scale/bias
operands, every tested boundary is bitwise exact:

- same-call JAX and separately materialized JAX FP32 invstd;
- materialized JAX and standalone real-Pallas FP32 invstd;
- Pallas interpret and real-Pallas FP32 invstd;
- materialized JAX and real-Pallas BF16-rounded invstd;
- explicit-invstd JAX and one-custom-call Pallas affine BF16 output.

Every comparison has zero mismatches, zero max/mean/RMSE and matching SHA-256
on all six corpora.  Real FP32 invstd, BF16 invstd and affine StableHLO modules
each contain one `tpu_custom_call`; interpret mode contains none.

Therefore rsqrt precision and affine arithmetic are not intrinsically inexact
in Pallas.  As with centered subtraction, drift appears only when producer and
consumer remain fused inside the larger LayerNorm kernel.  The next exact
baseline must explicitly materialize mean, variance and BF16 invstd between
Pallas calls, then execute affine/skip/ReLU from explicit operands.

The safe result is
[`artgor_layernorm_invstd.json`](artgor_layernorm_invstd/artgor_layernorm_invstd.json).
Private logs remain local.
