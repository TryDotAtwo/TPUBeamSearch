# External Stream3 split V1

Inspected 2026-09-05: COMPLETE and all three cases exact. Source
`c73f601bc2a1a0133535b18de6f9e816704ed350`, launcher `fc6a7f5`.
Eight physical TPU v5 lite devices (IDs 0..7), JAX/jaxlib 0.10.2,
libtpu 0.0.42.1. JSON contains input hashes and matching expected/output
SHA-256 for every case, including all five result arrays.

| Capacity/device | Mismatches | Compile s | Median ms | p10 / p90 ms |
|---|---:|---:|---:|---:|
| 256 | 0 | 1.56821 | 0.73925 | 0.72200 / 0.78174 |
| 512 | 0 | 1.42386 | 0.75874 | 0.72312 / 0.78149 |
| 1024 | 0 | 1.66508 | 0.82799 | 0.804281 / 0.88511 |

Fixtures include empty, full and partial counts; all-local, all-remote and
mixed owners. Independent host partition checks stable ordering, route packing,
full metadata including parent high words, local count, per-owner remote
counts, exclusive offsets and neutral tails. Owners are supplied, not derived
by the tested pipeline. The whole control reduction scratch compiles at these
three shapes; this does not establish legality at every supported API shape.

Each timing uses three warmups and 21 synchronized executions after exactness.
The measurements are diagnostic latencies, not matched A/B, not inference or
beam speedup, and not profiler evidence of overlap. Two external bitonic sorts
remain a correctness baseline. Large-frontier efficiency is unproven.

Preserved artifacts: `beam_external_split/external_split.json`, three compiled
HLO files and full Kaggle log. Next: compose real owner computation after
dedup, preserve score/payload winner, and validate collector/RDMA independently.
No per-shard cap is introduced. Full source parity and GPU/8-TPU replay remain
open; this gate is split-only.
