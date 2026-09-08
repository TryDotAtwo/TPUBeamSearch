# Routed response epoch gate

Status: V1 submitted, last checked QUEUED; physical acceptance pending. Source
8dc27ca763764c8485988c06210fddfc7c0f5854, launcher 1ec87f5. Builds on accepted
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

Source architecture recheck: ARCHITECTURE_NEED.md lines645-649 retain
selection outputs in a common prefix while selection temporaries overlay
materialization exchange temporaries. Line1688 places next_frontier_states_tmp
inside final scratch, not in an additional persistent allocation. Accordingly,
an integrated caller must account candidate bytes, accumulated target/validity
coverage, prepared grouped responses and exchange snapshots in materialization
peak lifetime, while retaining the common prefix. A standalone allocation per
helper would not establish the required one-pool architecture.

Current beam_scratch.plan_scratch only supplies geometry. Its read helper
copies data, and its write helper aliases only its own arena input/output;
neither proves whole-caller physical reuse. Require compiled memory/alias
evidence for the integrated caller before declaring the three overlays done.
make_final_coverage_agreement returns a common error, not a DMA-drained token:
publication must depend on completed transport, consumers and history work,
not solely on the common-error scalar being zero.

History ABI must remain distinct from request routing. Final requests encode
return rank in low16 and move in bits16..23. History projection retains the
original metadata route (meta plane7); reconstruct_history reads move in low8
and previous source rank in high16. Do not store request word3 as the history
route or replace that original source rank with the response destination.
RankHistoryStore.append_all_rank_layer validates all host layers before one
replacement assignment, but explicitly assumes completed transfers and a
successful distributed decision. Its atomic host update is not an atomic
frontier/history commit across TPU devices.

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

### Concrete integration points audited after submission

`beam_final_validation.pallas_validate_final_requests` currently enforces
`target_count.shape == (1,)` and evaluates `r[2] >= t[0]`; its documented
contract delegates return-rank validation to the caller. Its reason bits are
parent=1, target=2, move=4, local-slot=8, count-overflow=16. Preserve these
existing local-call semantics when adding the routed variant.

`beam_final_materialize.pallas_materialize_final` invokes that scalar validator
before `pallas_final_error_summary`; the DMA predicate is whole-batch
`errors[0,0] == 0`. Therefore a routed validator must feed this same pre-DMA
summary, not merely reject a response after materialization has occurred.
Use source-local `parents.shape[0]` only for the parent bound; it cannot supply
the target bound. Propagate the per-return-rank logical count table through
the remote snapshot materialization caller before enabling routed requests.

Regression requirements for that change: retain scalar callers unchanged;
mixed ranks with identical target index but different logical bounds;
zero-count destination; invalid rank and reserved byte; poisoned inactive
tails; count overflow; and a whole-batch rejection with zero output wire.
Assign any new reason bits explicitly without changing existing bit meanings.
The table access itself must be safe for malformed rank values, independently
of the final error mask. These are implementation requirements, not claims
that the routed validator or its TPU gate already exists.

## Launcher failure capture

Keep the launcher parent free of JAX imports. Run the probe in a child with
stdout/stderr redirected to a durable process.log under the Kaggle output
root. Write a pending process manifest before start, then record returncode
after termination, including native signal/abort failures. The child creates
its own fresh output subdirectory; parent must not pre-create that directory.
Do not infer success from the Kaggle terminal state: require child returncode
zero and independent report acceptance. Pin public source SHA and the already
accepted JAX/jaxlib0.10.2, libtpu0.0.42.1 runtime. Do not update or submit the
launcher until the current source regression is terminal and source published.
