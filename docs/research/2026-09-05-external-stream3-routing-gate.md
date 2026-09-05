# Next gate: external Stream3 routing and collector

## Established boundary

External dedup V3 is exact on eight TPU at capacities 256 and 512. Its return
ABI is metadata `[8,N]` and count `[1,128]` (lane zero only). It does not route.
The existing `pallas_stream3_split` accepts `[1]` counts and uses a whole-array
VMEM `_sort`. Direct composition would reintroduce the multi-vreg gather
limitation that motivated external sorting. Therefore do not treat the bounded
split as an HBM-scale implementation simply because its Python shape permits N.

Source contract checked against `D:/100XH100/ARCHITECTURE_NEED.md`, Stream3:
dedup raw Hash128 first, score/payload winner, then avalanche-derived owner,
then local/remote partition and owner-grouped remote records. Parent high bits
remain intact. Route is source rank/owner/move, not an input sorting key.

## Implementation sequence and red tests

1. Add an external split accepting compact dedup metadata, owners `[1,N]`
   and aligned count `[1,128]`. Owners are an explicit first gate input; deriving
   them from Hash128 is a separate composition test, never an implicit oracle.
   Prepare eleven planes in 128-column tiles; retain sorted-source index.
   Use external valid-first sorts `(valid,index)` for local and
   `(valid,owner,index)` for remote. Never use whole-N `_sort` here.
2. Per-tile counts must use `[1,tiles*128]` segments, not `(tiles,128)`
   with one-row blocks. Produce padded owner counts and exclusive offsets.
   Neutral tails remain invalid even with UINT32_MAX threshold.
3. Red tests: N256/512, empty/full, all-local/all-remote, uneven owners,
   owner ties, nonzero parent high word, stable within-owner ordering,
   counts and offsets including zero-count peers. Compare original C++ Stream3
   only after composing real owner computation with dedup; synthetic owners
   alone cannot establish source parity.
4. Compose dedup -> owner -> split using Pallas data-plane operations.
   Adapter tests must explicitly cover `[1,128]` versus `[1]` control boundaries.
   Match original source winner IDs and restored metadata before transport.
5. Collector is a separate acceptance boundary: preallocated resident A/B,
   write only to a nonprocessing sibling that fits; exact-full succeeds,
   neither sibling fitting sets fatal overflow. No truncation, spill or cap.
   Test repeated arrivals and lifetime/ack before buffer reuse.

## Physical bundle

After failing tests, minimal implementation, full local verification and public
source SHA, run one private session with independently retained case outputs.
First capacities 256/512, then 1024/4096 if compilation succeeds. Record each
compile rejection separately. Require eight real devices, runtime/source hashes,
zero mismatches and equal output hashes before three warmups/21 measurements.
Separate split-only, composed S3 and collector timings; do not infer a beam
speedup, overlap or large-N sorting efficiency from these correctness baselines.
Only then connect the previously validated variable-count RDMA wire protocol.

This document is a pending implementation protocol, not evidence that external
routing, owner integration or the collector is implemented or TPU-validated.
