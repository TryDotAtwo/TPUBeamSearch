# One-kernel residual-block A/B v2: BM256 VMEM rejection

Private Kaggle TPU v3-8 kernel `trydotatwo/tpu-layernorm-block-diagnostic`,
version 2, reproduced the earlier Dense and Dense+LayerNorm measurements, then
rejected the new one-kernel residual block at compile time.

The exact compiler result was `CompileTimeScopedVmemOom`: the
`BM=256, BK=256, BN=512` layout requested 16.31 MiB of scoped VMEM against the
16.00 MiB limit, exceeding it by 320 KiB. No one-kernel latency or correctness
result exists for this tile.

This is now handled by the comprehensive sweep rather than by hard-coding a
single replacement tile. It screens BM128 and BM256 across BK128/256,
BN256/512, BF16/FP32 statistics, and one/two-kernel boundaries. Compile and
VMEM rejections are recorded per candidate and do not terminate the run.
