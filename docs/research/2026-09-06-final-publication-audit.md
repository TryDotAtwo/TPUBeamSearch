# Final publication audit, pending fixes

Read-only CUDA source `D:/100XH100/cuda/dispatcher.cu:4634-4664` drains all
response slots, requires history and response totals equal local_target_count,
synchronizes stream3, schedules history copy, and only then copies temporary
next frontier into current frontier. These checks belong to the integrated
TPU caller too; successful isolated scatter is not a publication barrier.

## Confirmed local scatter defect

Current `pallas_scatter_final_responses` checks target bounds but does not
reject count greater than wire capacity. Direct interpreter reproduction:

- frontier uint8[1,128], all99;
- wire uint8[128,128], all0 (target zero);
- uint32 count129, logical state_len120;
- observed invalid_count0 and frontier_unchangedFalse.

No physical TPU claim. This is a malformed-batch guard defect, not proof of a
normal-workload failure. The current full-suite session83193 predates a fix
and is intentionally left unchanged. After it terminates: add a failing
regression requiring nonzero invalid count and unchanged frontier, implement
whole-batch count rejection before DMA, and run focused plus full verification.

Uniqueness/missing-target checks across chunks and coordinated publication are
still caller requirements. Count equality alone cannot detect a duplicate
paired with a missing target. The CUDA excerpt establishes total-count checks,
not a demonstrated per-target uniqueness check.

## Materializer boundary also trusts the caller

`beam_final_validation.py` explicitly delegates count<=capacity to its caller;
`pallas_materialize_final` does not add that check. Interpreter reproduction
with parents ones[1,128], 24 identity permutations, requests zeros[4,128],
count129 and target_count1 returns invalid_count0 and nonzero wire. This is
consistent with the low-level validator's documented precondition, but leaves
the integrated batch boundary dependent on external validation. Do not claim
the materializer itself rejects every malformed batch.

Next patch should guard both public materialization and scatter batch counts
before DMA. Preserve the source-compatible low four request reason bits; any
additional capacity reason must be explicitly documented as a TPU guard.
Tests must exercise count0, capacity, capacity+1 and UINT32_MAX. On overflow:
materialization produces zero wire; scatter preserves the entire frontier;
both expose an error. This does not replace cross-chunk count accounting,
target uniqueness or the final coordinated publication barrier.
