# Final exact cap and ordering contract

Read-only source audit, 2026-09-06. This is source evidence, not CUDA execution.

| Source | SHA256 |
|---|---|
| `D:/100XH100/cuda/threshold.cu` | `caa1e743369760a3616f0485dbd3b7b33484f23461e211755e730ebe91720975` |
| `D:/100XH100/cuda/final_materialize.cu` | `c3578668299fd2883ef9a0dda336e7bba1376f4ed44fad613a0867ad041585ea` |
| `D:/100XH100/cuda/dispatcher.cu` | `45ccfe9ddd27886acbb90ebb30ab67ef96a0557e2c7577043ac9aa3fc1b6027b` |

## Selection is two stable phases, not a score sort

`threshold.cu:131` marks only `slot < clean_count[shard]`. Phase0 is
`score_key < threshold`; phase1 is equality. Flattening is shard-major and
slot-major, including allocated holes which never become selected rows.
The 256-lane block scan and ordered block-prefix scan preserve this traversal.
The selected local buffer contains less-phase rows followed by equal-phase
rows, each stable in traversal order. Hash or parent sorting at this stage
would change the source contract.

`dispatcher.cu:4180` gathers per-rank less/equal counts and computes:

- `L = sum(less_counts)`, `E = sum(equal_counts)`;
- `K = min(effective_global_beam_width, L + E)`;
- `less_base[r] = sum(less_counts[:r])`;
- `equal_base[r] = L + sum(equal_counts[:r])`.

For a row's zero-based local phase ordinal `j`, global index is its phase
base plus `j`. The exact filter keeps only global indices `< K`.
The dispatcher subsequently requires local selected count to equal
`less_counts[r] + min(equal_counts[r], max(0, K-equal_base[r]))`.
Consequently the pipeline requires a threshold for which `L <= K`; silently
truncating an excessive less phase is not successful dispatcher equivalence.
Keep this invariant as an explicit failure gate in the TPU integration.

## Placement and provenance

`final_materialize.cu:97-156` maps global index `i` to
`target = floor(i*world/K)`, with target interval beginning at
`ceil(target*K/world)`. Local index is `i - begin`. Empty target intervals
are permitted when K is smaller than world; K=0 generates no requests.
Source rank comes from route bits16..31; owner bits8..15 are not the source.
Return rank is the balanced target. Parent index retains all64 bits.

Published `pallas_final_balance` and `pallas_final_plan` implement placement
given agreed boundaries and global indices, not upstream winner selection.
The caller still has to provide exact phase ordinals and collective counts.

## Next implementation and acceptance

1. Count/scan both phases over clean resident shards without flattening holes
   into valid candidates. Preserve traversal provenance through TPU layouts.
2. Exchange counts with all ranks, including empty ranks. Construct pair-word
   global prefixes and exact K; reject `L > K` before materialization.
3. Stable compact selected metadata and pair-word global indices into bounded
   output capacity; overflow is a failure, not silent truncation.
4. Compose the existing destination/request primitives; group requests by
   source while retaining return rank and target-local index.
5. Replay ties spanning rank and shard boundaries, empty ranks, K=0,
   underfilled beams, high parent words and phase-prefix carry. Compare the
   exact selected identities/order, not only count or score distributions.

Actual GPU/8-TPU multi-depth replay and transport/history acceptance remain
required. A local NumPy formula test does not establish CUDA equivalence.
