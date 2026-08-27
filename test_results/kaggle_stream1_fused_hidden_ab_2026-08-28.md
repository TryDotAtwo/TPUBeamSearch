# TPU Stream1 fused-hidden A/B

## Contract

- Network prefix: virtual one-hot `14400 -> 1536`, ReLU, then `1536 -> 512`, ReLU.
- Batch: 256 reachable Megaminx states.
- Checkpoint head: `MOVE_COUNT=24`.
- Inputs and weights: BF16; dot accumulation: FP32; layer boundaries: BF16.
- Timing: 10 warmups and median of 31 synchronized samples.
- Every production tile satisfies `BM % 8 == 0`, `BK % 128 == 0`, and `BN % 128 == 0`.

The separate variant uses two top-level Pallas calls and materializes `[256,1536]` BF16 `hidden1`. The fused variant uses one top-level Pallas call with two nested `emit_pipeline` stages; `hidden1` remains in VMEM.

## Fair A/B

Both variants were swept over the same aligned configurations. The stable selected configuration is:

`BM=256, BK_input=128, BN_input=512, BK_hidden=256, BN_hidden=512`.

| Kaggle version | Separate | Fused | Fused speedup | Saved |
|---|---:|---:|---:|---:|
| v2 | 0.443571 ms | 0.432040 ms | 1.0267x | 11.531 us |
| v3 | 0.465870 ms | 0.455791 ms | 1.0221x | 10.079 us |

For the best configuration selected independently within each run:

| Kaggle version | Best separate | Best fused | Speedup | Saved |
|---|---:|---:|---:|---:|
| v2 | 0.443410 ms | 0.432040 ms | 1.0263x | 11.370 us |
| v3 | 0.465870 ms | 0.455339 ms | 1.0231x | 10.531 us |

All paired fused outputs matched their same-tiling separate outputs exactly (`max_abs_error=0`). Both runs checked out Git commit `06dd7ea5b53bf52d3166acb7f7d7ef11a823c9b0` before execution.

## Decision

Keep the fused kernel. Its reproducible gain is modest but positive: roughly 2.2-2.7%, or 10-12 microseconds for this batch and network prefix. Use the stable selected tile above as the default. The earlier apparent 1.55x gain was not a valid fusion estimate because it compared fused `(BM=256, BN_input=512)` against separate `(BM=128, BN_input=256)`; most of that difference came from tiling.

Raw outputs and logs are stored locally under `test_results/kaggle_stream1_fused_hidden_v2/` and `test_results/kaggle_stream1_fused_hidden_v3/`.
