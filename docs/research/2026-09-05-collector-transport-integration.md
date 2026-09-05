# Collector/transport integration boundary

Current physical S3 evidence ends at local/remote split. The tested variable
exchange factory is now extracted into `beam_remote_exchange.py`
(`make_variable_exchange_call`); the integration harness remains in
`benchmarks/beam_rdma_ring_probe.py` (`make_integrated_stream3_exchange`).
The extraction preserves the existing snapshot/ack behavior; its ABI trace
passes locally, but the extracted module still needs physical confirmation.
`beam_dma_ring.py` is a host lifetime oracle, not the device transport.

Before integrating the existing physical exchange:

1. Extract its existing behavior without changing semaphore ordering; retain
   the same fixtures and compare the extracted version on physical TPU.
2. The receive slot remains owned until its consumer finishes reading it.
   Existing code copies each slot into a distinct `2+epoch` snapshot before
   ack, so a later collector can consume that independent snapshot after slot
   reuse. Removing snapshots requires collector completion before slot ack.
   Producing a reservation or next-count tensor alone is not consumption.
3. Route received and local records to logical shards using the separate shard
   hash salt. Owner rank is not a logical-shard identifier.
4. Reserve a complete shard group in one nonprocessing sibling. Tile size is
   not group size and cannot relax capacity or split admission across siblings.
5. Publish counts only after all record stores are complete; acknowledge the
   receive slot only after its consumer has finished reading it. A zero-count
   peer participates in the epoch without starting a nonexistent copy.
6. On fatal capacity failure, stop through a coordinated protocol without
   silently dropping records or leaving peers waiting for an acknowledgement.

Acceptance must include repeated slot reuse, empty ranks, uneven groups,
exact-full and overflow, then simultaneous S4 processing of the other sibling.
Functional copies with whole-tuple synchronization are a reference baseline,
not evidence of resident aliases or overlap. Memory reports and traces must
establish actual lifetimes before claiming either property.

Source audit: `make_variable_exchange_call` currently gates payload DMA on
nonzero counts but copies the full fixed-capacity record block for nonzero
peers. It retains all epoch snapshots as well as two reusable receive slots.
Therefore variable counts do not establish count-proportional network bytes,
and the snapshot baseline does not establish bounded two-slot-only storage.
Preserve that behavior on extraction; measure and optimize it separately.

## Hash-to-shard grouping baseline

`pallas_collector_partition` groups supplied logical shard IDs stably, returns
counts and exclusive offsets, and neutralizes invalid metadata tails.
`pallas_collector_hash_partition` supplies the actual independent Hash128 shard
salt through `pallas_route_hashes`, without rewriting source/owner/move words.
The external-sort baseline accepts power-of-two capacity 128..16384 and up to
256 logical shards. It assumes valid counts/IDs and is not an efficient resident
scatter implementation; full-width scratch reduction remains a scaling limit.

TDD: missing partition entrypoint failed before implementation; three supplied-ID
interpreter cases passed (mixed four shards, empty three shards, all in one of
three shards). The actual hash composition separately failed on its missing
entrypoint, then passed in 15.56 s. Its independent Python uint64 reference
checks stable metadata, counts and offsets for 213 valid records out of 256,
three shards, and distinguishes the shard salt from the owner salt.
Neither test is a TPU compilation or latency result. Full regression finished:
654 passed in 466.33 s, no skips, with the source CPU oracle enabled; saved XML
is `test_results/local_collector_routing_regression.xml`. The all-shard preflight
below was added after that suite's collection and has separate targeted evidence.

## CUDA admission scope audit

Read-only `D:/100XH100/cuda/stream3.cu`, SHA-256
`935d8ca7b81982281cafa8b2aa2218b6cb87471dad6e564535bb9dea687bb1fd`:
the partition launcher sorts the complete `max_candidates` input by shard,
then `ReduceByKey` produces one `raw_count` per valid shard (lines 1366..1392).
`stream3_prepare_partition_counts_kernel` (line 557) checks all groups before
`stream3_partition_scatter_kernel` (line 726) launches on the same CUDA stream.
The scatter returns immediately for a nonzero shared fatal flag (line 750).
Therefore one overflowing shard prevents **all metadata writes and dirty-count
increments in this partition invocation**, not only that shard's writes.
The preparation phase can still change write-buffer selection/diagnostic
scratch before failure; this is not a blanket rollback of every control field.

The multi-shard TPU integration must preflight every group and aggregate fatal
before scattering any group. Looping over the current single-shard append API
and discovering a later overflow after an earlier append would not preserve
this CUDA boundary. Tiles must not subdivide admission groups. Distinct input
invocations remain distinct admissions; this audit does not combine all epochs.

CUDA dirty-count increments occur inside the scatter kernel; the inspected
function does not itself demonstrate concurrent S4 publication. Same-stream
completion or external event ordering must be inspected before modeling overlap.
No CUDA execution or concurrent-read correctness is claimed from this audit.

`pallas_collector_preflight` now implements read-only all-shard admission.
Each plan contains sibling, destination offset, full amount and enabled flag;
any overflowing group or prior fatal zeros every plan and sets one fatal flag.
Two interpreter tests were red before implementation and now pass (6.89 s):
exact-full admission with sibling selection, and late-shard overflow cancelling
an earlier otherwise valid group. Metadata scatter and dirty publication are
not yet attached. Full control-field rollback is not asserted against CUDA.

The next functional stage is now attached by `pallas_collector_scatter` and
`pallas_collect`: actual Hash128 shard grouping -> all-shard preflight ->
metadata scatter -> next control values. The caller must await all returned
arrays before exposing controls. This is not an in-place resident protocol:
buffers are functional outputs and the full grouped input is a VMEM window
for each destination tile. That deliberately transparent baseline needs
physical compilation, then tiled DMA/alias work before production scaling.

TDD evidence for this attachment: three scatter tests failed on the missing
entrypoint and pass in 5.75 s (unaligned source/destination, late overflow,
empty/busy input). The full hash collector test failed before implementation
and passes in 25.74 s: three shards fill A, then B, then reject a third complete
input without metadata/dirty mutations. Independent Python uint64 hashing
determines expected membership. These are interpreter results only; full
regression for this attachment plus the gather correction and physical harness
completed: 664 passed in 435.19 s with source CPU oracle enabled, no skips.
Saved XML: `test_results/local_collector_bundle_regression.xml`.

## Dispatcher publication and receive admission boundary

Read-only `cuda/dispatcher.cu` SHA-256
`45ccfe9ddd27886acbb90ebb30ab67ef96a0557e2c7577043ac9aa3fc1b6027b`:
the normal local ring records `stream3_done` after its graph; host completion
queries/synchronizes that event before exposing its ready queue (3552,
3638..3668). The inspected remote path synchronizes stream5 payload exchange,
then calls `stream3_collect_remote_recv_cuda` once with the complete
`recv_total_64`, synchronizes stream3, checks fatal, and appends its ready queue
(3186..3240). This is evidence of a completion boundary, not lock-free dirty
counter publication from inside the scatter kernel.

Consequently TPU per-peer epoch snapshots must be compacted into one receive
partition for the corresponding exchange round before admission. Calling
collector separately on each snapshot would change group sizes, A/B choices
and fatal-overflow behavior relative to this source. Local-input admission is
still a separate invocation. Preserve explicit batch boundaries in replay.

The source ready-queue kernel marks the selected physical shard processing
before enqueueing it and redirects collector writes to another writable
sibling (`cuda/stream3.cu`, 903..930). The host also guards at most one running
S4 job per logical shard. A TPU resident implementation needs equivalent
exclusive ownership and a complete-write publication boundary; the current
functional full-tuple wait is only the serialized reference version.
