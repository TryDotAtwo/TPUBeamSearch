# Full LayerNorm ResMLP v1: correctness failure

Private Kaggle TPU v3-8 run `trydotatwo/tpu-layernorm-full-mlp`, version 1,
tested the exact Artgor `q555_2k_BEST.pt` checkpoint at local batch 16,384.

The original `jax_model.apply` reached 1.393M states/s. Both Pallas variants
were finite, but neither was correctness-valid:

| implementation | states/s | max abs | mean abs | argmax agreement |
|---|---:|---:|---:|---:|
| original JAX | 1,393,159 | - | - | - |
| Pallas separate | 632,198 | 0.40625 | 0.20469 | 0% |
| Pallas per-layer fusion | 611,036 | 0.40625 | 0.20469 | 0% |

Root cause: Artgor's BF16 inference keeps LayerNorm statistics and affine
operations in BF16, while v1 deliberately used FP32 LayerNorm statistics.
The small per-layer numeric difference accumulated across 21 LayerNorms. The
benchmark also mislabeled finite outputs as `valid`; this is fixed by an
explicit exact-output and argmax correctness gate.

The corrected v2 candidate reproduces Artgor's BF16 LayerNorm arithmetic.
Its performance is not inferred from this failed run and must be measured on
TPU before scaling work starts.
