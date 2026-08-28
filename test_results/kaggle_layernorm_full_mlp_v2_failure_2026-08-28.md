# Full LayerNorm ResMLP v2: invalid benchmark input exposed

Private Kaggle TPU v3-8 run `trydotatwo/tpu-layernorm-full-mlp`, version 2,
reproduced Artgor's BF16 LayerNorm arithmetic. Both Pallas variants remained
finite but failed the full-model gate on the benchmark input:

| implementation | states/s | max abs | mean abs | argmax agreement |
|---|---:|---:|---:|---:|
| original JAX | 1,384,467 | - | - | - |
| Pallas separate | 524,788 | 0.359375 | 0.18802 | 0% |
| Pallas per-layer fusion | 495,671 | 0.359375 | 0.18802 | 0% |

The run also exposed a benchmark defect: every row was the same `0..149`
sequence, although the checkpoint embedding accepts only categorical values
`0..23`. Thus the comparison used out-of-domain gather indices and measured
argmax agreement for only one invalid state repeated 16,384 times.

Version 3 replaces it with deterministic, diverse states whose every value is
within `[0, NUM_CLASSES)`. The numerical gate allows normal BF16/MXU reduction
differences (`max_abs <= 0.5`) but still requires at least 99% argmax agreement.
No scaling result will be accepted until this representative correctness gate
passes.
