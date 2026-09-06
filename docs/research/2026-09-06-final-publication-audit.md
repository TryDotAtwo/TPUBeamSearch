# Final publication audit and remaining integration gates

## Current status (supersedes the original reproduction notes below)

The batch-capacity defects below were fixed in published commit `f9a8bee`.
`test_beam_final_count_guard.py` verifies count129/UINT32_MAX rejects the
entire batch, zeroes materializer output and preserves the scatter frontier;
count0/128 remain valid. The full snapshot passed896 tests with both C++
oracles (`test_results/local_v9_full.xml`). This is local evidence, not
physical TPU acceptance. The original reproduction is retained for attribution.

Target coverage and collective rejection are implemented separately; see
`2026-09-06-final-coverage-gate.md`. They do not implement the remaining
integrated transaction. Its required order is:

1. Finish request, response and history exchanges, including empty peers.
2. Validate complete response and history target sets and accumulated errors;
   all ranks participate in the common decision even after a local error.
3. On rejection, retain current frontier and do not publish the new depth.
4. On acceptance, finish temporary-frontier writes and preserve history input
   until the host copy consumer has completed.
5. Publish the next frontier, then permit scratch-layout reuse only after all
   consumers of the common prefix have drained.

The source dispatcher schedules history copy before frontier copy; it does
not imply that host history copy completes at that point. A TPU implementation
must retain that distinction rather than treating a scheduled copy as a drain.
The present host history store publishes rank layers independently and requires
caller coordination. It is not yet a transactional multi-rank publication path.

Read-only caller follow-up: `tools/production_runner.cu:2689-2725` queries
`slot.copy_done` (or synchronizes with `wait_all`) before starting the host
history writer. At lines4830-4887 the depth caller passes that event into
finalization, commits the slot and calls `pump_completed(false)`. The explicit
synchronizations there are conditional debug/verbose branches. Thus host
writer readiness is demonstrated by source; an unconditional device scratch
reuse dependency is not established by these excerpts. Do not describe this
as a proven CUDA race: the complete lifetime/alias path needs separate tracing.
For TPU acceptance, require an explicit completed consumer dependency before
overwriting shared device history input, and separately wait for host writer
completion before reusing its host slot or reconstructing an unpublished layer.

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

## Scratch transition must preserve a common prefix

Read-only `ARCHITECTURE_NEED.md:638-650` requires one static pool with three
exclusive layouts. Selection outputs consumed by materialization occupy a
common prefix: a phase change must not overwrite them with exchange temporaries.
Selection filter temporaries and materialization exchange temporaries can alias
only outside that live prefix. Persistent stream survivor/count/histogram
storage begins after the reserved final budget. Current frontier and solved/
stop storage are outside the pool.

`ARCHITECTURE_NEED.md:1588-1617` requires clearing response padding, scattering
to temporary next frontier, finishing all responses, copying into current
frontier and only then releasing final storage for streams. The TPU caller
must additionally await every consumer of common-prefix history data before
reuse. Python reference replacement does not establish this lifetime, nor
does a local scatter completion establish a multi-rank drain.

Acceptance evidence remains missing: actual donated/aliased HBM buffer report,
per-phase high-water sizes, and a full-depth profile proving no overlap of
incompatible layouts. The current separate-array primitives do not establish
the required single-pool implementation.
