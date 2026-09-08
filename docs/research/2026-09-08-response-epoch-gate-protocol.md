# Routed response epoch gate

Status: protocol; physical run not prepared or launched. Builds on accepted
local scatter V9 and exchange V7 without repeating those isolated gates.

## Scope and oracle

Compile preparation (wire, packed requests, receive control, materialization
validation) and `make_final_response_chunk_call` in one eight-rank shard_map.
Use runtime inputs, not captured payload constants. Keep prepared grouped
buffers alive across three calls with dynamic chunk indices 0, 1, 2.
The call returns private bytes and count/error controls, not a frontier commit.

Generate distinct bytes for every source, destination, original slot and byte
position. Build expected results on the host from original request records:
for each source independently, filter valid requests by return rank preserving
original order; take [128*epoch:128*(epoch+1)], concatenate in source-rank
order for each receiver, then zero-pad to 1024 rows. Do not use production
grouping, interval, packing or compaction helpers for this oracle.

## Cases

- Empty all ranks, poisoned invalid request/payload tails.
- Self-only; cyclic destination (source+1)%8; one-to-all; all-to-one.
- Uneven per-peer counts with 0, 1, 127, 128, 129; use enough per-source
  allocated rows for every peer's selected count. Three identical epochs
  on all ranks, even when one rank has exhausted its responses.
- One source with materialization failure and zero send intervals: every
  receiver must report error and no bytes/counts in every epoch.
- One source with malformed return rank or reserved byte; one with receive
  error. Same common error expectation, even on otherwise healthy ranks.
- Repeat the valid cases with different bytes after failed cases, verifying
  semaphores and snapshot storage cannot leak previous payloads or errors.

Gate every output byte and all control lanes on all eight ranks against the
independent oracle; preserve SHA, exact mismatch counts and finite integer
ABI metadata. Pending case records must be written before compile/execution.
Save source SHA, input hashes, eight unique device IDs/kinds, JAX/jaxlib/libtpu,
lowered MLIR, compiled HLO, nested process log and partial JSON on failure.
Coordinator acceptance requires terminal child success AND every expected
case/epoch independently validated. A native abort is not a source diagnosis.

## Boundaries still required afterward

This tests real routed response transport but not parent request generation,
materialization against remote source parents, destination-specific capacity,
duplicate/missing target coverage, history or atomic final publication.
Do not scatter returned bytes until receive error has been collectively gated.
For multi-chunk assembly keep a private candidate frontier and coverage across
all epochs; a late error invalidates the entire candidate, including earlier
valid chunks. Publish only after final global validation and DMA drain.
Performance and overlap require separate profiling of the integrated caller;
this correctness gate does not establish either.

## Existing coverage/scatter integration constraint

`pallas_final_target_coverage` expects all local targets across chunks and
proves the sorted live sequence is exactly range(target_count). Checking each
chunk against the final target count would incorrectly reject partial chunks;
checking only total received count would miss a duplicate replacing a missing
target. Preserve the target/validity ledger across every epoch.

`pallas_scatter_final_responses` checks bounds but requires unique targets;
it does not establish uniqueness itself. Do not feed it an unvalidated chunk
with duplicate targets. A first integrated correctness implementation can
stage complete responses, validate whole-final coverage and common errors,
then scatter into the private frontier. A streaming implementation instead
needs an explicit incremental duplicate gate before each scatter plus final
missing-target validation. Neither path may reuse the published old frontier
as scratch. Account full-response staging in the scratch budget if choosing
the former; it is not free memory or an established production layout.

## Destination capacity audit

The existing request validator accepts scalar target_count and compares every
request target against that scalar. After requests from different return ranks
are compacted together, this is insufficient for unequal destination counts.
Neither the source parent's frontier size nor the maximum destination count
is the appropriate logical bound for every request. Before integrating remote
materialization, validate each request against the count belonging to its
low16 return rank (and validate that rank before indexing any count table).
Allocated storage capacity and logical selected count must remain distinct.
Receiver-side final coverage remains necessary even after this source check.
Include a fixture where target 1 is legal at one destination but illegal at
another, plus zero-count destinations and poisoned invalid tail ranks.
