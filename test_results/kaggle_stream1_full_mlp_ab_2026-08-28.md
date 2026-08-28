# TPU Stream1 full-MLP head-fusion A/B

## Contract

- MLP: virtual one-hot `14400 -> 1536`, ReLU, `1536 -> 512`, ReLU, `512 -> MOVE_COUNT`.
- `MOVE_COUNT=24`; output head has no ReLU.
- Batch: 256 reachable Megaminx states.
- Inputs, weights, and layer boundaries: BF16; dot accumulation: FP32.
- Prefix tile: `BM=256, BK_input=128, BN_input=512, BK_hidden=256, BN_hidden=512`.
- Head sweep: `BK_output={128,256,512}`, `BN_output={128,256}`.
- All production tile dimensions satisfy the TPU alignment validator.

The separate-head variant uses the selected two-layer fused kernel followed by one aligned Pallas dense head. The full-fusion variant adds a third nested `emit_pipeline`; `[BM,512]` remains in VMEM. Physical output is padded, while the public result is always exactly `[batch,MOVE_COUNT]`.

## Correctness and layout

- Kaggle TPU accepted both physical `BN_output=128` and `BN_output=256`.
- Every paired configuration produced exactly the same 24 BF16 logits: `max_abs_error=0`.
- Local suite: 12 tests passed, including a hand-derived three-layer fixture that checks logical output trimming and absence of output ReLU.

## Order-neutral A/B

Versions 3 and 4 used 10 warmups and 31 synchronized samples per variant. Measurements alternated `AB/BA` inside each same-tiling pair to neutralize execution-order drift.

| Run | Best separate head | Best full fusion | Best-to-best result |
|---|---:|---:|---:|
| v3 | 0.420280 ms | 0.423170 ms | fusion 0.683% slower |
| v4 | 0.455360 ms | 0.454480 ms | fusion 0.194% faster |

The absolute difference changed sign and stayed between -2.89 and +0.88 microseconds. This is not a reproducible performance gain.

## Decision

Use two stages for the production Stream1 MLP:

1. Fused virtual-one-hot `14400 -> 1536 -> 512`, keeping the first hidden boundary in VMEM.
2. Separate `512 -> MOVE_COUNT` Pallas head, with conservative tile `BM=256, BK=512, BN=256`.

Keep `pallas_fused_mlp` as a correct experimental implementation, but do not select it as the production path. The separate head has equivalent measured performance, less VMEM scratch and simpler scheduling. Full fusion can be revisited only if a later Stream1 stage consumes logits inside the same kernel.

The order-neutral runs checked out Git commit `0990dda270f373439a16bf061dfd8859ac9af683`. Raw JSON and logs are stored locally under `test_results/kaggle_stream1_full_mlp_v3/` and `test_results/kaggle_stream1_full_mlp_v4/`.
