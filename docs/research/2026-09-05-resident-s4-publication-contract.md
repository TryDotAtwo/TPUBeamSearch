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

## External S4 record path

`pallas_external_stream4_dedup` shares the HBM-staged sort/unique/compact
machinery, selecting Hash128, score, parent high/low and route keys instead of
S3 payload tie-break. The original S3 API retains its keys. Capacity remains
bounded to16384 by this baseline; no top-k is applied.
The missing-entrypoint test failed first; the new S4 tie test plus existing
external dedup tests pass together (5 tests, 46.46 s). Subsequent original CPU
C++ comparisons cover empty and duplicate-heavy partial input with non-neutral
unused storage. All three S4 tests pass in47.46 s. This is not CUDA/TPU
execution; histogram commit, resident writes and physical composition remain.

## Histogram and explicit DMA commit experiments

The bounded score-sort/range-search histogram handles valid count, score-domain
exclusion and aligned zero padding. Three interpreter cases pass in24.64 s,
including empty and one-run-across-all-tiles inputs. Source SCORE_BIN_COUNT is
307201; the implementation avoids an NxBins comparison matrix but retains a
whole sorted-score search window and needs physical sizing/performance tests.

`pallas_commit_histogram` aliases histogram A/B and publication control outputs.
It copies each inactive histogram tile through VMEM with explicit DMA waits,
then flips active and clears processing in a final control DMA which is also
waited. Two missing-module tests failed before implementation; both A/B
directions pass TPU interpreter race detection (2 tests, 4.13 s), preserving
the previously active histogram. This requires prior clean-record completion
and caller exclusion of concurrent inactive readers/writers. It is not a
coordinated S5 snapshot or physical alias/memory-report proof. The physical
collector V2 remains on its older pinned source, unaffected by these changes.

The stronger `pallas_commit_s4` also writes clean metadata before histogram DMA
and only then publishes clean/dirty/processing/active control. Its initial
missing-module test was red; a three-tile histogram/two-tile metadata case
passes race detection (7.20 s). `pallas_run_reserved_s4` composes count capture,
external S4 dedup, matching histogram and that commit without CPU count reads.
Its red-before-implementation test passes in26.52 s, checking clean/dirty
duplicates, threshold rejection, exact output histogram and inactive-slot flip.
Both were added after full-regression12711 collection; they have separate
evidence only. Reservation/queue integration, sibling concurrency, scratch
preallocation, physical alias reports and complete TPU execution remain open.

The reserved-job test now reuses returned resident/histogram versions for a
second job with an empty result: it verifies flip-back, clearing the newly
inactive histogram and preservation of the former active histogram. This
expanded test passes in45.44 s. Reservation is explicitly simulated in the
test; it is not a device ready-queue implementation or concurrent S5 proof.

## Local committed snapshot arithmetic

`pallas_sum_committed_histograms` selects each physical shard's active A/B
histogram and sums into low/high uint32 planes with explicit carry. Its test
failed before the module existed, then passed in5.59 s: eight UINT32_MAX
contributions give low=0xfffffff8/high=7, while inactive poison is excluded.
The control ABI currently permits128 physical shards. The caller must freeze
selected histogram versions through snapshot completion; independent active
loads are not a concurrency protocol. Global reduction, threshold request
coordination and monotonic threshold publication remain separate missing work.

Periodic threshold arithmetic was checked against the source branch contract
in `cuda/threshold.cu` (SHA256
`caa1e743369760a3616f0485dbd3b7b33484f23461e211755e730ebe91720975`).
The Pallas tiled carry-prefix implementation finds the first inclusive prefix
at least beam. Four tests pass in8.27 s after a shape-broadcast correction,
covering first initialization, insufficient total and non-relaxation of an
initialized threshold. This returns a candidate publication value only; it
does not itself implement S5 request reduction or active-slot publication.

## Regression checkpoint

The combined local regression completed with712 passed in1033.87 s; artifact:
`test_results/local_collector_v3_regression.xml`. This includes the S4 job,
snapshot/threshold arithmetic and collector aligned-offset correction.
The later recovery coordinator has six separate passing tests (0.27 s),
including missing/malformed reports and native-abort return codes retaining
partial logs without promoting the integrated gate. This is local evidence,
not physical TPU confirmation. Collector V3 will first retry the full collector
and only then run integrated S3 in a sequential child process if full is exact.
