# Final transport integration boundary

Current source inspection: `beam_final_group.py` groups arbitrary uint32
payload planes by rank and original ordinal. `beam_remote_exchange.py`
transports fixed eight-plane S3 records and retains per-peer snapshots.
`test_beam_final_logical_replay.py` still filters and routes on the host.
Neither grouping nor host replay proves a distributed final implementation.

## Required wire contracts

| Path | Payload | Destination | Consumer |
|---|---|---|---|
| Request | parent lo/hi, target-local index, return-rank/move (4 uint32 planes) | original source rank | immutable current frontier + generator materialization |
| Response | STATE_LEN state bytes + little-endian uint32 target-local index, aligned padding | balanced return rank | bounds/uniqueness gate then frontier scatter |
| History | parent lo/hi, original route, target-local index (4 uint32 planes; validity is control) | balanced destination rank | rank-local completed layer |

The five-plane history projection contains four payload planes plus validity;
validity is not an additional history field. The original route is not rewritten
to the destination rank. Parent high words are never discarded.

## Next code sequence

1. Produce per-rank start/count from the sorted validity/rank planes in Pallas.
   Reject a live out-of-range rank; ignore invalid padding. Empty rank count is
   zero, with a defined exclusive-prefix start. Counts sum to live records.
2. Pack bounded peer chunks from these intervals. Chunk count is collectively
   agreed so empty senders still enter every control epoch. Local-self records
   use the same consumer validation but do not need remote DMA.
3. Add a separate parameterized final transport factory; do not silently alter
   the accepted fixed-eight-plane S3 factory. Make HBM output allocation explicit
   and validate compiled output layouts on the target runtime.
4. Preserve count preflight, readiness, wait-send, wait-receive and consume-ACK.
   A zero-count peer completes control without waiting for nonexistent payload.
   No send-source reuse before send completion; no destination reuse before
   its consumer has copied or finished reading it.
5. Complete all request chunks before releasing current frontier. Complete all
   response/history chunks and reject duplicate/missing target indices before
   publishing the next frontier/history layer. A failure is collective; never
   skip a peer handshake because a local validator failed.

## Acceptance fixtures

Use all eight ranks with all-empty, self-only, one-to-all, all-to-one and uneven
traffic; counts 0/1/127/128/129 and multiple slot wraps. Match exact selected
identities, parent/source/move provenance, destination index and reconstructed
states. Include invalid ranks, high parent words, duplicate targets, capacity
overflow and a failed rank while other ranks have traffic. First validate
transport independently, then integrate selection and multi-depth history.

The existing snapshot transport is serialized and capacity-proportional for
nonzero peers. Reuse of its protocol does not establish count-proportional
traffic, direct-to-consumer execution or overlap. Measure those separately.
