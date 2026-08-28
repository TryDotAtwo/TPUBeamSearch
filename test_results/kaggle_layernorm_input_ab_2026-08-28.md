# Cube555 LayerNorm input encoding A/B

Private Kaggle kernel: `trydotatwo/tpu-layernorm-input-ab`, version 2.
Source commit: `76f593b15c13ba1a39f4c6f958dfde6aaf4d0cd8`.
Checkpoint: `q555_2k_BEST.pt`. Runtime: JAX 0.10.2 on TPU v3-8.
The timed prefix is categorical encoding, `3600 -> 1024` dense, LayerNorm,
and ReLU for a local batch of 16,384. Compilation is excluded.

| Encoding | BM | Median | States/s | Relative to original | Result |
|---|---:|---:|---:|---:|---|
| Original JAX embedding gather | — | 7.021 ms | 2.334M | 1.000x | oracle |
| Pallas embedding gather | 256 | 8.369 ms | 1.958M | 0.839x | valid; max abs 0.0078125 |
| Pallas virtual one-hot MXU | 256 | 11.166 ms | 1.467M | 0.629x | valid; max abs 0.0078125 |
| Pallas fused virtual one-hot | 256 | — | — | — | Mosaic compile rejection |
| Pallas candidates | 1024 | — | — | — | scoped VMEM rejection |

The ordinary embedding gather is the selected input encoding for this model
family. Replacing each lookup with a virtual one-hot MXU multiplication is
33.3% slower than the Pallas gather path and 37.1% slower than original JAX.
This differs from the previous direct categorical MLP because Artgor's model
already compresses every 150-way category to a 24-wide embedding before the
large dense layer.

The pre-folded virtual-one-hot candidate is not accepted: Mosaic rejects the
required mapping with `Not implemented: Multiple source vregs along gather
dimension`. This is a compiler/layout rejection, not a valid timing result.

The Pallas gather path remains slower than original JAX because this experiment
uses a standalone Pallas LayerNorm, already measured at 0.5x XLA speed. The
next decision is therefore the correctness-equivalent fused
`Dense -> LayerNorm` kernel, not further one-hot work.

Safe raw results are stored in
`test_results/kaggle_layernorm_input_ab_v2/stream1_layernorm_input_ab.json`.
The full private Kaggle log remains local.
