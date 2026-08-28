# Full LayerNorm ResMLP v3: representative correctness result

Private Kaggle TPU v3-8 run `trydotatwo/tpu-layernorm-full-mlp`, version 3,
used 16,384 deterministic diverse states with all 150 categorical values in
the checkpoint's valid `[0, 150)` embedding domain. The checkpoint embedding
shape is `150 x 24`: 150 categorical values and embedding width 24.

| implementation | states/s | versus JAX | max abs | mean abs | argmax agreement |
|---|---:|---:|---:|---:|---:|
| original `jax_model.apply` | 1,395,949 | 1.000x | - | - | reference |
| Pallas separate | 526,228 | 0.377x | 1.3125 | 0.17708 | 71.67% |
| Pallas per-layer fusion | 494,233 | 0.354x | 1.3125 | 0.17708 | 71.67% |

All outputs were finite, but both Pallas variants failed the representative
correctness gate (`max_abs <= 0.5`, argmax agreement at least 99%). They are
also 2.65-2.82x slower than the original JAX implementation. Fusion does not
change Pallas output, so the remaining discrepancy is not caused by the fusion
boundary; it comes from accumulated BF16/MXU reduction-order differences in
the custom dense/LayerNorm path.

Therefore the selected implementation for this exact LayerNorm checkpoint is
the original JAX/XLA `jax_model.apply`. The next scaling run measures it on
valid diverse states at 1 TPU, 8 TPU, and the real 128-chunk scan. This is an
evidence-based fallback, not a claim that Pallas cannot help a differently
trained or numerically tolerant model.
