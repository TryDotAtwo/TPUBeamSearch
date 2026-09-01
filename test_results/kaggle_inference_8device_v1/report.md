# Exact eight-device inference v1 report

## Result

This run did **not** pass the exact-Q promotion gate.  It nevertheless found a
large verified throughput opportunity: prepacked FP32 banks with Pallas BM2048
ran1.617x faster than original JAX on the eight-device legal corpus and1.669x
faster on categorical stress.  The rejection is numerical, not performance:
12 legal Q values and55 stress Q values differed from the original BF16 output.

No32K confirmation or performance profile was run because the frozen16K exact
gate rejected every optimized encoding arm.

## Reproducibility

- Private Kaggle kernel: `trydotatwo/tpu-exact-inference-8-device`, version1.
- Public source: `d2159cb230ef77deeb5a4a2b6a42181a62dc027c`.
- Python3.12.13, JAX/jaxlib0.10.2, libtpu0.0.42.1, NumPy2.5.0.
- Eight visible and active `TPU v5 lite` devices in one process.
- Checkpoint SHA256:
  `2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2`.
- Original model source SHA256:
  `6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e`.
- Puzzle SHA256:
  `01c616cb943d574d1b63109f350b30c7710656e53e2a5eaebb5d50ed0e495ff0`.
- Legal/stress full input SHA256:
  `340cce563b0b84eff52fc13da7979be261297e948223db5830fb0c7df267a743` /
  `8d054a06a9fcfa339cc77cecab7b98546f5212462c009403955ee390bff53955`.
- Seeds: legal42, categorical stress43.  Every contiguous device-shard hash is
  recorded in the source JSON.

The measured executable is only the complete150x24 embedding -> Dense3600x1024
-> ten-block1024-wide LayerNorm ResMLP ->30-score minimizing-Q forward.  It
contains no move expansion, top-k, deduplication or other beam-search work.

## Synchronized steady inference

Local batch is16,384 on every active device.  Compile, first execution, bank
preparation and host conversion are excluded.  Twelve timed rounds follow five
warmups and alternate forward/reverse case order.

| Devices | Corpus | Variant | Median ms | Global Mstate/s | Exact Q |
|---:|---|---|---:|---:|---:|
| 1 | legal | original JAX | 11.5193 | 1.422 | yes |
| 1 | legal | Pallas prepacked BM2048 FP32 | 6.6627 | 2.459 | yes |
| 1 | stress | original JAX | 11.4161 | 1.435 | yes |
| 1 | stress | Pallas prepacked BM2048 FP32 | 6.6775 | 2.454 | yes |
| 8 | legal | original JAX | 12.1696 | 10.771 | yes |
| 8 | legal | Pallas prepacked BM2048 FP32 | 7.5274 | 17.413 | **no:12** |
| 8 | stress | original JAX | 12.2072 | 10.737 | yes |
| 8 | stress | Pallas prepacked BM2048 FP32 | 7.3140 | 17.921 | **no:55** |

The corresponding fixed-local-work 1-to-8 throughput factors are7.57/7.48
for original JAX and7.08/7.30 for BM2048 on legal/stress, or parallel
efficiencies0.95/0.94 and0.89/0.91.  These scaling numbers describe rejected
candidate timing and are not an accepted model result.

BM scaling is monotonic over the tested FP32 banks.  On eight-device legal,
BM64/128/256/512/1024/2048 take11.85/9.27/8.52/7.73/7.61/7.53ms; stress takes
11.76/9.17/8.37/7.65/7.50/7.31ms.  Runtime banking BM128 takes9.84/9.70ms and
tiled JAX10.67/10.56ms.

## Correctness localization from v1

Original runtime JAX and the typed BF16 JAX control are elementwise identical
on both one- and eight-device samples.  On eight devices, tiled JAX, runtime
Pallas BM128 and every successful prepacked FP32 BM produce one common output
hash per corpus:

- legal candidate:
  `07e68ced5cec6d6b50d781678dff0dcee43ee796a5f66dca6bfbbb73bed57784`;
- legal original:
  `2d9a71658d04fc4a41f3da665c3aa7cfcc15f2094ede0599fceac9780b534827`.

The12 legal mismatches occur only on global rows29,807 and50,224.  Maximum
absolute error is0.25; argmin remains identical.  Stress has55 differing Q
values, maximum error0.125 in the retained witness row29,369, and one changed
argmin among131,072 states (agreement0.99999237).

This does not yet prove that an eight-device lowering is causal.  The one-core
screen used only global rows0..16,383, so it never evaluated these witnesses.
The next experiment therefore replays their exact owner shards at the same
local batch on one core before changing execution APIs.

Compiled HLO gives a narrower negative result: typed reference and tiled JAX
use the same BF16 operand layouts for the first3600x1024 Dense, the same
`EmitOutputBatchInLanesKernelOutputFeatureInLanes` convolution emitter and the
same window configuration.  That evidence rules out a simple tile/emitter
explanation but does not establish equal producer bits or equal later fusion.

## Storage and failed arms

The original runtime model passes99.03MB of dynamic arguments; typed BF16 uses
49.51MB.  FP32 prepacked banks contain393,216 bytes and the full banked runtime
tree is49.90MB.  Bank construction is a one-time model-init action outside
steady timing.

At eight-device legal, compiler-reported static temporary storage is723.66MiB
for original JAX,723.63MiB for typed JAX,123.97MiB for tiled/runtime Pallas and
123.88MiB for prepacked BM2048.  These are static allocation estimates, not
hardware bandwidth, cache, utilization or live-memory counters.

All six BF16-bank BMs fail Mosaic compilation on both corpora.  The retained
error is `tpu.dynamic_gather`: indices are i32 while the gathered result is
BF16, and that different-bitwidth combination is not implemented in this
runtime.  FP32 physical banks compile; their values were first rounded to the
same logical BF16 checkpoint embedding.

## Next experiment

The published execution A/B at source
`88d6e42c4100578aa9478d3faf6b4f5d30adc01f` tests the witness-owning one-core
shards, every inference boundary, input barriers, a true two-dispatch
device-resident Pallas-to-JAX pipeline, direct sharded `jit`, `pmap` and eight
independent one-partition executables.  Promotion still requires exact full Q
and a speed win over the fastest exact JAX control on both corpora at16K and
32K local batch.

Raw JSON, full Kaggle log, StableHLO and compiled HLO are retained beside this
report.  No result in v1 changes BN behavior or a production default.
