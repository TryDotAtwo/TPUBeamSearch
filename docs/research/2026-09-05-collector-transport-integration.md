# Collector/transport integration boundary

Current physical S3 evidence ends at local/remote split. The tested variable
exchange implementation currently resides in
`benchmarks/beam_rdma_ring_probe.py` (`make_variable_exchange_call` and
`make_integrated_stream3_exchange`), not in the production collector module.
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
