# Stream1 first-layer Pallas A/B — Kaggle TPU v3-8

Kernel: `trydotatwo/tpu-stream1-first-layer-a-b`, version 7, COMPLETE.

Git source: `TryDotAtwo/TPUBeamSearch@9e79e2331a6c97caa3b9259be598d1530b4cc6da`.

Verified contract from the attached MOVE_COUNT-head checkpoint:

- `MOVE_COUNT=24`
- `STATE_LEN=120`
- `STATE_STORAGE_LEN=128`
- `NUM_CLASSES=120`
- first-layer weight shape `[14400, 1536]`
- output head shape `[24, 512]`

Both Pallas implementations use the same BF16 folded BatchNorm weights and FP32 accumulation:

1. `virtual_one_hot_mxu`: construct one-hot K tiles in VMEM and multiply on MXU.
2. `embedding_sum_vpu`: dynamically fetch selected embedding rows and sum them.

Single-core steady-state medians:

| Parent batch | Virtual one-hot MXU | Embedding sum | MXU speedup |
|---:|---:|---:|---:|
| 32 | 0.566 ms | 4.551 ms | 8.0x |
| 128 | 0.613 ms | 16.950 ms | 27.6x |
| 256 | 0.787 ms | 33.305 ms | 42.3x |

Correctness:

- maximum cross-implementation error: `2.98e-8`
- maximum error against the independent folded-input reference: `2.38e-7`

Decision: use virtual one-hot tiles in VMEM with tiled MXU matmul for the Stream1 first layer. Do not materialize the `[batch, 14400]` input in HBM. The direct embedding-sum path is retained as a correctness/reference implementation, not the production hot path.

Raw local evidence: `test_results/kaggle_stream1_ab_v7/stream1_first_layer_ab.json` and `test_results/kaggle_stream1_ab_v7/tpu-stream1-first-layer-a-b.log`.
