# Stream1 optimized scaling — Kaggle TPU v5e-8

Kaggle kernel `trydotatwo/tpu-stream1-optimized-scaling`, version 1,
completed using Git commit `6879edb09fcc8146430aa58907a34c361c410eb7`.
JAX 0.10.2 reported eight TPU devices.

## Fixed inference contract

The run used the complete `120→1536→512→2×residual(512)→24` MLP with
`STATE_STORAGE_LEN=128` and the selected prefix tiling:

```text
local batch = 32768 states/device
BM = 1024
BK_input = 128
BN_input = 1536
BK_hidden = 256
BN_hidden = 512
residual fusion = separate
pipeline = Pallas default
```

Weights were replicated, state batches were independently sharded, and no
collective was used in the measured inference path.

## Correctness

At batch 256, the optimized Pallas result had maximum absolute BF16 drift
`0.125` against the JAX reference. This passes the established `≤0.25` gate.
Every measured output was finite.

## Scaling

| TPU devices | Global batch | Median | States/s | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 32,768 | 9.880 ms | 3.317 M | 1.000× | 100.00% |
| 2 | 65,536 | 9.998 ms | 6.555 M | 1.976× | 98.82% |
| 4 | 131,072 | 10.075 ms | 13.009 M | 3.922× | 98.06% |
| 8 | 262,144 | 10.287 ms | **25.482 M** | **7.683×** | **96.04%** |

The eight-device result corresponds to approximately 611.6 million logits/s
because every state produces `MOVE_COUNT=24` logits.

Using the earlier logical estimate of 47.93 MFLOP/state, the measured aggregate
rate is approximately 1.221 PFLOP/s. This is timing-derived, not a TPU hardware
counter measurement.

## Decision

The optimized prefix preserves near-linear independent-shard scaling. Use local
batch 32768 and global batch 262144 for eight-device Stream1 integration. The
remaining 3.96% scaling loss is small relative to the next beam-search stages;
further Stream1-only scaling work is not currently justified.
