# Composed external Stream3 V1

COMPLETE and exact at N256/512 on eight TPU v5 lite devices, IDs 0..7.
Source `bb6c38a9b8be1c2678e6136a316382a9831fc458`, launcher `9619cdc`.
JAX/jaxlib 0.10.2; libtpu 0.0.42.1. Inspected 2026-09-05.

| Capacity/device | Mismatches | Compile s | Median ms | p10 / p90 ms |
|---|---:|---:|---:|---:|
| 256 | 0 | 2.66243 | 0.77270 | 0.723859 / 0.821509 |
| 512 | 0 | 2.63779 | 0.83023 | 0.78282 / 0.86087 |

All five outputs match saved original CPU C++ Stream3 outputs, with equal
combined SHA-256: local/remote metadata, local count, peer counts and offsets.
The composed path performs inclusive threshold, raw Hash128 dedup with
score/payload winner, actual owner computation, then stable split. Fixtures
cover empty/full/partial ranks, duplicate/hash-zero inputs, reverse payload
ties, high parent words and UINT32_MAX threshold. Neutral tails are compared.

Manifest and archive hashes match the published fixtures. The C++ checkout
was dirty when fixtures were generated: its commit alone is NOT a complete
provenance identifier. The manifest retains actual source/header, adapter and
rebuilt executable hashes. This proves parity with that recorded CPU source
snapshot, not execution of CUDA or any unrecorded checkout state.

Each exact case has three warmups and 21 synchronized samples. These are
diagnostic latencies, not matched comparisons with separate calls, prior
probes, GPU beam or Artgor notebook. Lower latency than sums of earlier
standalone probes does not prove overlap or fusion benefit without matched
controls/profile. Four bitonic sorts remain a correctness baseline.

Boundary: metadata already corresponds to payload. Ring payload restoration,
collector, RDMA, resident S4, S5, final materialization and full-depth replay
are not covered. Next acceptance is collector/transport integration with
capacity and lifetime failures, not a change to shard-local selection policy.

JSON, full Kaggle log and both compiled HLO files are retained alongside this
report. No full beam speedup or complete architecture claim is made.
