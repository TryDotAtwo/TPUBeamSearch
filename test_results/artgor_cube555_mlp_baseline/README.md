# Artgor Cube555 LayerNorm MLP baseline

Measured on 2026-08-28 with private Kaggle kernel
`trydotatwo/artgor-cube555-tpu-mlp-benchmark`, source commit `a659d7f`, JAX
0.10.2, and eight TPU v3 cores. Compilation is excluded from steady-state
timing; each timed call is synchronized.

## Exact model

- Checkpoint: `q555_2k_BEST.pt`
- Parameters: 24,757,807
- State: 150 categorical `uint8` values
- Encoding: shared `150 x 24` embedding
- Trunk: `3600 -> 1024`, LayerNorm, ReLU
- Residual stack: 10 blocks, each containing two `1024 -> 1024` dense layers
  and two LayerNorm operations
- Q head: `1024 -> 30`
- Inference dtype: BF16

## Results

| Execution | Local batch | Global batch | Median latency | Throughput |
|---|---:|---:|---:|---:|
| 1 TPU | 16,384 | 16,384 | 11.848 ms | 1.3828M states/s |
| 1 TPU | 32,768 | 32,768 | 24.559 ms | 1.3342M states/s |
| 8 TPU, one chunk | 16,384 | 131,072 | 12.382 ms | 10.5859M states/s |
| 8 TPU, real 128-chunk scan | 2,097,152 | 16,777,216 | 1.4457 s | 11.6049M states/s |

Eight-TPU parallel efficiency at the matched 16,384 local batch is 95.69%:

```text
10.585918M / (8 * 1.382807M) = 0.9569
```

The real scan is 9.63% faster in aggregate than separately timing one chunk,
which is consistent with amortized dispatch/pipelining effects. It does not
change the model arithmetic.

The dense arithmetic is approximately 49.377 MFLOP/state, close to the existing
BN Stream1 model's 47.93 MFLOP/state. At the 16,384 single-core result this is
about 68.27 timing-derived TFLOP/s. The large throughput gap versus the existing
BN model therefore cannot be explained by dense FLOP count alone; the next
experiments isolate input encoding and the twenty runtime LayerNorm operations.

Raw measurements are in `artgor_cube555_mlp_benchmark.json`. The downloaded full
Kaggle log is intentionally retained locally rather than published.
