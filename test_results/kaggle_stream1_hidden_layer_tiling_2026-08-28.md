# TPU Stream1 `1536 -> 512` tiling

## Reproducibility

- Kaggle kernel: `trydotatwo/tpu-stream1-hidden-layer-tiling`, version 2
- Git commit: `532d434a13a756e2bc311ae568b6ad3b2602a50d`
- Backend: Kaggle TPU, JAX 0.10.2
- Checkpoint output: `MOVE_COUNT=24`
- Batch: 256
- Timing: 10 warmups, median of 31 synchronized samples

## Alignment contract

- `MXU_DIM=128`
- `BM % 8 == 0`
- `BK % 128 == 0`
- `BN % 128 == 0`
- Logical dimensions are padded separately; invalid production tiles are rejected before Pallas compilation.
- Tested logical matrix: `[256,1536] @ [1536,512]`.

## Result

| Tile `(BM,BK,BN)` | Median |
|---|---:|
| `(128,128,512)` | **0.15465 ms** |
| `(256,256,256)` | 0.15712 ms |
| `(256,512,256)` | 0.16539 ms |
| `(256,256,512)` | 0.16540 ms |
| `(128,256,512)` | 0.16705 ms |

The current baseline is `(128,128,512)`. The next result is only about 1.6% slower, so this is a measured choice for the present shape and TPU, not a universal optimum.

The Pallas output matches the BF16 JAX reference exactly for this input (`max_abs_error=0`). Bias and ReLU are fused into the final K step.

## First-layer boundary and fusion decision

- First virtual-one-hot/MXU layer: 0.71274 ms.
- Two-layer JIT pipeline: 0.69739 ms.
- Materialized `hidden1`: `[256,1536]` BF16.
- HBM write plus read: 1,572,864 bytes (1.5 MiB).
- Roundtrip time estimated at 412.5 GB/s peak bandwidth: 3.813 microseconds.
- That optimistic transfer-time estimate is 0.547% of the measured pipeline.

This estimate suggests that fusion may be a small win, but it is not a mathematical upper bound: effective HBM bandwidth and synchronization can increase the real boundary cost. Keep `(128,128,512)` as the separate-kernel baseline, then compare it with a genuinely fused Pallas implementation before making the fusion decision.

Raw JSON and Kaggle log were downloaded locally under `test_results/kaggle_stream1_hidden_v2/`.
