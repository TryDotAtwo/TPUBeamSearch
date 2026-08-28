# TPU Dense + LayerNorm fusion A/B

Private Kaggle kernel: `trydotatwo/tpu-layernorm-fusion-ab`, version 1.
Source commit: `22c013a446822cc93ff167607d3c0aaaa4416e1d`.
Runtime: JAX 0.10.2 on TPU v3-8. The measured layer is BF16
`1024 -> 1024` dense followed by FP32-reduction LayerNorm at local batch
16,384. Medians use 15 synchronized iterations after five warmups.

| Mode | BM/BK/BN | Median | States/s | Fusion speedup | Max difference |
|---|---|---:|---:|---:|---:|
| Separate | 128/256/512 | 1.473 ms | 11.125M | — | oracle |
| Fused | 128/256/512 | 1.262 ms | 12.981M | 1.1668x | 0 |
| Separate | 256/256/512 | 1.180 ms | 13.884M | — | oracle |
| Fused | 256/256/512 | 1.137 ms | 14.408M | 1.0377x | 0 |

Fusion is accepted because both configurations match the separate Pallas path
bit-for-bit. It removes the HBM dense-output boundary and the standalone
LayerNorm launch.

The production candidate for the full LayerNorm ResMLP is per-layer fusion with
`BM=256`, `BK=256`, and `BN=512`: it has the highest absolute throughput. The
larger relative gain at BM=128 does not compensate for its smaller dense tile.

Safe raw results are stored in
`test_results/kaggle_layernorm_fusion_ab_v1/stream1_layernorm_fusion_ab.json`.
The full private Kaggle log remains local.
