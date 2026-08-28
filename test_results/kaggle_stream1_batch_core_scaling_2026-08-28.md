# TPU Stream1 batch and core scaling

Kaggle kernel `trydotatwo/tpu-stream1-batch-core-scaling` completed versions
1 through 3. The final extended sweep used commit
`2d96c294e20cc60d8307ebf0486cc768fd780a4d`; the corrected sharding helper
was validated on commit `e24a0e53bc6653354ec10cc6d300113bad32f0d2`.

The complete output-24 residual MLP used production
`residual_fusion="separate"`. Weights were replicated; states and logits were
sharded across a one-dimensional TPU mesh. No collective is included in these
inference timings.

## Single-core batch and BM sweep

The initial sweep covered batch 64 through 4096 with `BM={128,256,512}`.
Small batches favor the smallest tile that avoids padding. From batch 512
onward, `BM=512` is fastest.

| Local batch | BM | Median | Throughput | Time/state |
|---:|---:|---:|---:|---:|
| 4096 | 512 | 2.1475 ms | 1.907 M/s | 524.3 ns |
| 8192 | 512 | 3.8808 ms | 2.111 M/s | 473.7 ns |
| 16384 | 512 | 7.4809 ms | 2.190 M/s | 456.6 ns |
| 32768 | 512 | 14.6731 ms | 2.233 M/s | 447.8 ns |
| 65536 | 512 | 29.2012 ms | 2.244 M/s | 445.6 ns |

Throughput is near its observed plateau by local batch 32768. Moving from
32768 to 65536 adds only 0.50% throughput while doubling latency.

All configurations through batch 4096 passed the BF16 reference gate with
maximum absolute error 0.125. Larger batches execute the identical static
kernel and were checked for finite logits without materializing the reference
gather intermediates.

## Independent-shard scaling

With local batch 65536 and `BM=512`:

| TPU cores | Global batch | Median | Throughput | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 65536 | 29.2498 ms | 2.241 M/s | 0.998x | 99.83% |
| 2 | 131072 | 29.3987 ms | 4.458 M/s | 1.987x | 99.33% |
| 4 | 262144 | 29.6211 ms | 8.850 M/s | 3.943x | 98.58% |
| 8 | 524288 | 29.9026 ms | 17.533 M/s | 7.812x | 97.65% |

At local batch 4096, the earlier eight-core run reached 13.158 M/s with
85.45% efficiency. Raising the local batch amortizes fixed dispatch/sharding
cost and recovers nearly linear scaling.

## Decision

- Keep `BM=512` for large production shards.
- Use local batch 32768 when latency matters; it is within 0.50% of the maximum
  measured per-core throughput.
- Use local batch 65536 when aggregate throughput is the only objective.
- Treat `17.53 M states/s` as the Stream1-only eight-core ceiling for this
  checkpoint and batch, not as complete beam-search throughput. Later stages
  must measure their collectives and communication separately.

Version 1 exposed a JAX 0.10.2 `shard_map` VMA validation requirement. The
production helper now uses explicit `out_specs` with `check_vma=False`; a local
regression test exercises the same independent-state/replicated-weight layout.
