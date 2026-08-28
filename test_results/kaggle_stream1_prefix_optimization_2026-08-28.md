# Stream1 prefix optimization — Kaggle TPU v5e-8

Kaggle kernel `trydotatwo/tpu-stream1-prefix-optimization`, version 3,
completed using Git commit `976072d1ff1564d320d4f916998e78b3f7f9b296`.
JAX reported version 0.10.2, TPU backend, and eight visible devices.

## Scope

The sweep kept the complete 24-output MLP semantics fixed. It varied the fused
virtual-one-hot `14400→1536→512` prefix over aligned `BM`, `BK_input`, and
`BN_input` tiles. It also compared the default Pallas pipeline with explicit
one-buffer, two-buffer, and two-buffer-plus-lookahead input schedules.

Weights were passed as runtime JIT arguments. Versions 1 and 2 accidentally
closed the weights over the benchmark lambda, causing XLA to treat about 44 MiB
of weights as compile-time constants and fail with a 46.19 MiB scoped VMEM
allocation against the 16 MiB limit. This was a benchmark defect, not a defect
in the production inference API.

## Prefix screen at batch 4096

| Configuration `(BM,BK,BN)` | Prefix time | States/s |
|---|---:|---:|
| `(1024,256,1536)` | 1.453 ms | 2.819 M |
| `(1024,128,1536)` | 1.458 ms | 2.809 M |
| `(1024,256,768)` | 1.521 ms | 2.693 M |
| old `(256,128,512)` | 2.799 ms | 1.463 M |

`(1024,512,1536)` was the sole rejected candidate: its scoped VMEM allocation
exceeded the TPU limit. The best accepted prefix configuration is therefore
`BM=1024, BK_input=256, BN_input=1536`.

## Pipeline A/B at old tiling

| Input pipeline | Prefix time | States/s | Relative to default |
|---|---:|---:|---:|
| default | 2.799 ms | 1.463 M | 1.000x |
| explicit 1 buffer | 2.774 ms | 1.476 M | 1.009x |
| explicit 2 buffers | 2.840 ms | 1.442 M | 0.986x |
| 2 buffers + lookahead | 2.961 ms | 1.383 M | 0.945x |

There is no evidence that missing manual double buffering was the bottleneck.
The default schedule already overlaps transfers adequately for this regular
GEMM. Lookahead adds overhead and is measurably slower. The large improvement
comes from doing substantially more useful work per row/output program and
reducing per-block scheduling and pipeline-bubble costs.

## Startup and steady state

For the winning prefix, throughput rose from 1.706 M states/s at batch 1024 to
2.879 M at 4096, 3.498 M at 32768, and 3.613 M at 65536. Thus small batches are
strongly startup dominated. From 32768 to 65536 only 3.3% additional prefix
throughput remains to amortize.

For the complete MLP, throughput was already flat:

| Configuration | Batch 32768 | Batch 65536 |
|---|---:|---:|
| `(1024,256,1536)` | 3.292 M states/s | 3.288 M states/s |
| `(1024,128,1536)` | **3.313 M states/s** | 3.274 M states/s |
| `(1024,256,768)` | 3.094 M states/s | 3.074 M states/s |
| old sweep baseline `(256,128,512)` | 1.605 M states/s | 1.603 M states/s |

The prefix-only winner and full-model winner are statistically very close. For
the complete model at the latency-oriented production batch 32768,
`BM=1024, BK_input=128, BN_input=1536` had the best median: 9.892 ms and
3.313 M states/s. It also compiled faster than the `BK_input=256` alternative.

Compared with the previous one-core frontier of about 2.244 M states/s, the new
full-model result is about 1.476x, or 47.6% faster. With the previously used
logical 47.93 MFLOP/state estimate, this corresponds to approximately 158.8
TFLOP/s per TPU device; this is timing-derived rather than a hardware-counter
measurement.

## Decision

Use the default Pallas pipeline and update the production prefix tiling to:

```text
BM=1024
BK_input=128
BN_input=1536
BK_hidden=256
BN_hidden=512
```

Keep local batch 32768 for latency. Before claiming eight-device aggregate
throughput, rerun the independent-shard scaling benchmark with the new tiling.
