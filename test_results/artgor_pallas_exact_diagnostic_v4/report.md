# Split-mean all-Pallas diagnostic v4

Private Kaggle kernel
[`trydotatwo/tpu-artgor-all-pallas-exact-diagnostic`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-all-pallas-exact-diagnostic),
version 4, completed on 2026-09-02 from source
`8d9ce0aa1dd0a4d6c1eaba59ef077b58d636767e`.

The run is a valid rejection on eight TPU v5 lite devices.  It evaluated the
single BK128 candidate with a dedicated Pallas mean dispatch followed by a
Pallas LayerNorm remainder dispatch.  Embedding and input Dense remain exact,
but the first semantic mismatch is still `input.layernorm_relu` on every
corpus.

The mismatch counts and errors are unchanged from the unsplit v3 result:

| Corpus | BF16 mismatches / 2,097,152 | Max abs |
|---|---:|---:|
| legal42 | 163,288 | 0.03125 |
| legal142 | 168,930 | 0.03125 |
| legal242 | 194,400 | 0.03125 |
| stress43 | 154,177 | 0.015625 |
| stress143 | 162,741 | 0.03125 |
| stress243 | 162,114 | 0.015625 |

Combined with the exact fixed-operand subtraction experiment, this proves the
centered FP32 drift is erased by the subsequent BF16 variance rounding and is
not responsible for the final mismatch.  The next causal boundary is the
already observed exact BF16 variance feeding epsilon/add/rsqrt/BF16 invstd,
followed by affine arithmetic.

No candidate reached full-stage exactness, so the benchmark correctly skipped
full-output timing and HLO promotion audit.  The safe result is
[`artgor_pallas_exact_diagnostic.json`](artgor_pallas_exact_diagnostic/artgor_pallas_exact_diagnostic.json).
Raw private logs are retained locally only.
