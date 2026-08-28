# TPU Pallas LayerNorm tiling

Private Kaggle kernel: `trydotatwo/tpu-pallas-layernorm-tiling`, version 2.
Source commit: `2dc195d9edff82245b2d619f52cf535cf18eaa62`.
Runtime: JAX 0.10.2 on TPU v3-8. Local batch 16,384, width 1,024,
BF16 input/output and FP32 Pallas reductions. Compilation was excluded; medians
use 21 synchronized iterations after five warmups.

| Implementation | BM | Median | Rows/s | Relative to JAX | Correctness |
|---|---:|---:|---:|---:|---|
| Original JAX/XLA | — | 0.3910 ms | 41.900M | 1.000x | oracle |
| Pallas | 128 | 0.7818 ms | 20.956M | 0.500x | valid, max abs 0.015625 |
| Pallas | 256 | 0.7819 ms | 20.954M | 0.500x | valid, max abs 0.015625 |
| Pallas | 512 | — | — | — | rejected: 21.8 MB scoped VMEM > 16 MB |
| Pallas | 1024 | — | — | — | rejected: 43.8 MB scoped VMEM > 16 MB |

The standalone Pallas LayerNorm is almost exactly two times slower than the
original fused XLA expression. `BM=128` and `BM=256` are statistically
indistinguishable. Increasing BM cannot recover performance because 512 and
1024 exceed the per-kernel scoped VMEM limit.

Therefore standalone Pallas LayerNorm is retained only as a correctness
baseline. The optimization path must fuse dense computation, LayerNorm, skip,
and ReLU while the row tile remains in VMEM. The separate-vs-fused benchmark is
implemented in `benchmarks/stream1_layernorm_fusion_ab.py`.

Safe raw results are stored in
`test_results/kaggle_layernorm_tiling_v2/stream1_layernorm_tiling.json`. The
downloaded full Kaggle log remains local and is not published.
