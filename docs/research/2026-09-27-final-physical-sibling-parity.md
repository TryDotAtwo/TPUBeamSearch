# Final physical-sibling parity audit

Read-only source: `D:/100XH100`, HEAD
`b5fcf6b0ee3247a1f189e6da79a1641d2304bd1c`. The inspected dispatcher,
Stream4 and architecture files have no local modifications. No CUDA run or
source edit was performed for this audit.

## Evidence and correction

The earlier TPU architecture sentence claiming final cross-A/B dedup was
incorrect for the current CUDA implementation:

1. `cuda/dispatcher.cu:1560` builds one S4 graph per physical storage shard,
   passing `survivor_shard + shard*capacity`, matching count pointers and one
   physical capacity.
2. `cuda/stream4.cu:472` thresholds, sorts and reduces that single physical
   buffer. It does not receive its sibling's data.
3. `cuda/dispatcher.cu:3754` force-readies and drains these jobs during final
   flush. Repeating an individual-buffer job does not merge A/B.
4. `cuda/threshold.cu:633` performs final less/equal count/scan/scatter over
   all storage shards; this path does not compare candidate hashes.
5. `cuda/stream3.cu:615` maps logical shard and buffer to
   `logical_shard*shard_buffer_count + buffer`.

The multigpu_beam expert independently pointed to these same calls and the
absence of cross-sibling dedup. That advice was checked against local source,
not treated as accelerator evidence.

For exact source parity, preserve physical order A0,B0,A1,B1, slot order,
phase less-than before equal-to-threshold, and separate rank prefixes per
phase. Identical hashes remaining in different physical buffers can count as
two final records. Their parent/history metadata must not be merged. Stronger
global uniqueness is a separate semantic change, not a TPU optimization.

## Implementation and evidence boundary

`pallas_final_resident_view` copies physical prefixes into that traversal and
rejects dirty, busy, fatal, over-capacity or prior-error state. It does not
prove DMA completion, histogram freshness or actual physical aliasing.

The two distinguishing fixtures use duplicate hash 7 in A0/B0 with parents
10/20. At equal score and cap4 both survive; at cap1 parent10 wins by physical
traversal. They run the real Pallas view, phase masks, scan, rank-prefix, cap
and final-index kernels in the interpreter. Seven new tests plus existing
phase/index tests pass (9 total, 15.53 seconds).

This proves local TPU algorithm behavior for supplied resident fixtures,
not reachability in an actual CUDA run or full multi-depth GPU/TPU parity.
Those target-hardware gates remain mandatory.
