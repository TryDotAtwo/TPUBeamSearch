# Resident S4 and histogram publication contract

Read-only source audit, not implementation or hardware acceptance.

Source: `D:/100XH100/ARCHITECTURE_NEED.md` lines 49..83, 1290..1330,
1799..1885; `cuda/stream4.cu` SHA-256
`3ef2d5d3318718b4284540d3d4988e3a9196070829417f6a8cada17b58d9a2a0`.

## Two independent double buffers

Each logical shard owns two physical survivor buffers for the entire depth.
Each physical shard also owns two histogram publication buffers. These are
different indices and lifetimes: choosing survivor A does not select histogram
A. Physical capacity is independent of the dirty-work trigger. At most one
physical sibling of a logical shard may be processing at a time.

An S4 job consumes the selected physical buffer's complete clean+dirty prefix
and a captured threshold. It filters once, sorts/deduplicates by Hash128, and
retains the minimum score, then parent_idx, then route_packed. It does not cap
the shard to any top-k and does not repeat thresholding with a newer threshold.
The sibling remains collector-owned if writable; no spill or dropped overflow.

## Actual CUDA publication sequence

`stream4.cu:243..265` writes compact survivors, clean_count and dirty_count=0.
This is not job completion: processing_flag remains set during histogram work.
The same stream then clears the inactive histogram, builds score/count pairs,
sorts/reduces them and writes the inactive histogram (`:270..337`). Only
`stream4_finalize_score_histogram_kernel` (`:101..108`, launched at `:338`)
flips histogram active index and clears processing_flag. A TPU port must not
release the physical buffer merely because dedup finished.

Histogram entries count clean records with score_key < SCORE_BIN_COUNT; a
score outside that domain is not a histogram bin (`:62..79`). Do not silently
clamp scores into the last bin. Intermediate CUDA reduction counts are uint64;
the physical-shard histogram stores uint32. Global sums need separate overflow
reasoning and cannot inherit the local counter width without a proven bound.

## S5 boundary

The May 26 contract supersedes the older local periodic collective description.
For multiple ranks, S4 never independently starts a collective. At S5 exchange
points every rank reduces its request flag; if any request is set, every rank
snapshots committed histogram selections and participates in the global sum.
Only then is a non-relaxing threshold published through its own inactive slot
and active-index commit. Empty ranks participate. Reset processed-work counters
at that coordinated boundary, not after independent local job counts.

## Implementation/verification consequences

1. Separate launch reservation, record completion, histogram completion and
   release. Test that a processing sibling blocks a second job and redirects
   collector writes without modifying the processing input.
2. Verify survivor identities/ties against original-source oracle, including
   clean+dirty duplicates, threshold capture, empty/exact-full and no shard cap.
3. Test histogram inactive clearing, exact counts and active selection across
   repeated jobs, including empty output and alternating publication slots.
4. Establish actual store-before-publication dependencies on TPU. Independent
   functional tuple leaves do not by themselves authorize concurrent consumers.
5. Verify resident buffer aliases, scratch reuse and DMA drain in compiled
   memory reports/profiles. Full-buffer functional outputs are only a semantic
   baseline, not evidence of preallocated resident execution.
6. Exercise coordinated S5 with uneven work, empty ranks and pending fatal/stop
   so no rank exits a collective epoch unilaterally.

Existing bounded dedup and collector gates do not satisfy these publication,
residency, global scheduling or end-to-end acceptance requirements.

## Normal ready selection: initial implementation

The live `stream3_build_ready_shard_queue_kernel` adds concrete scheduling
details beyond the architecture summary. It computes a reserve from ceil(S3
batch/logical shards), adds a quarter rounded up, caps it at capacity, and uses
capacity-reserve as the clean-ready threshold. A dirty buffer is eligible at
the dirty trigger OR near capacity. A nonempty clean-only buffer is eligible
near capacity. Dirty-ready takes priority, then available sibling space; ties
normally prefer non-current, with a full-capacity current exception.

`beam_s4_ready.pallas_claim_ready` implements this normal two-sibling selection
with the derived clean-ready threshold passed explicitly. Six interpreter
cases were red before implementation and pass in 6.44 s: dirty threshold,
equal-space tie, clean-only near capacity, below-trigger idle, processing
sibling exclusion, and dirty over clean priority. It produces a serialized
reservation and job descriptor, not a resident job or compacted global queue.
Force-dirty/force-clean drain modes were subsequently added after five failing
tests (missing arguments). All eleven selection/drain cases pass in 8.38 s.
Force-clean does not flush a dirty-only buffer; force-dirty does not claim a
clean-only buffer below its normal trigger, and neither claims an empty one.
The static derived-threshold helper was added after five missing-function
failures; all sixteen current tests pass in 8.36 s. It rounds average shard
write and its quarter upward, caps reserve at capacity, and uses wide host
arithmetic to avoid wrap in configuration round-up. It reads no hot-path
candidate counts. Full source differential coverage and physical lowering
remain to be added.
These tests were added after the running full regression's collection and
are not covered by that run.

Additional checks cover both-full current-buffer ties, reversed current index,
busy B, and unequal sibling capacity. The complete selection test file now
passes 21 cases in 8.95 s (zero skips/failures/errors), with XML at
`test_results/local_s4_ready_regression.xml`. This remains interpreter/host
evidence and is explicitly separate from the earlier full-suite snapshot.
