# TPU Stream1 residual-fusion A/B/C

Kaggle kernel `trydotatwo/tpu-stream1-residual-fusion-a-b-c`, versions 2
and 3, completed on commit `a47e490c56ae12269de9df58d01f2b8eb6254027`.

## Variants

- `separate`: two ordinary aligned Pallas dense calls per residual block.
- `per_block`: one Pallas call contains both `512 -> 512` matmuls, skip add,
  and ReLUs; branch and skip stay in VMEM during the block.
- `pairs`: one Pallas call contains both complete residual blocks; the value
  between blocks also stays in VMEM.

All measurements cover the complete model from aligned states to 24 logits,
not an isolated residual microbenchmark. Batch is 256; each variant receives
10 warmups and 31 samples. Execution order rotates between A/B/C.

## Results

| Run | Separate | Per block | Pair | Winner |
|---|---:|---:|---:|---|
| v2 | 0.475380 ms | 0.473960 ms | 0.474811 ms | per block by 0.300% |
| v3 | 0.463060 ms | 0.464011 ms | 0.464170 ms | separate by 0.205% |

Compile plus first execution in v2 was 0.401 s for separate, 0.447 s for
per-block fusion, and 0.633 s for pair fusion. V3 reproduced the same compile
ordering: 0.391 s, 0.447 s, and 0.650 s.

Both fused variants had maximum absolute BF16 drift 0.125 and mean drift
0.0104726 relative to separate kernels. Argmax agreement was 95.3125% on the
256 diagnostic states. CPU Pallas interpret tests match hand-derived residual
fixtures exactly; real Mosaic pipelines use a different MXU accumulation order.

## Decision

Keep `residual_fusion="separate"` as the production default. The measured
sub-microsecond difference changes sign across identical runs, so neither
fusion has a reproducible throughput advantage. Pair fusion also increases
compile latency materially. Retain both fused implementations for future
experiments with wider/deeper residual stacks, where avoided HBM traffic may
become large enough to measure.
