# Exact target coverage before final publication

`pallas_final_target_coverage` verifies that the complete local final target
set is exactly `range(target_count)`. It sorts `(valid,target)` through the
existing external HBM bitonic primitive and checks both the expected valid
prefix and each index. Capacity is power-of-two and >=128, below signed32.
Any nonzero validity is true; padding values are not targets.

This catches duplicate+missing pairs even when aggregate receive counts match.
It must consume all local targets across chunks, not merely the current chunk.
The current diagnostic implementation costs O(N log^2 N) HBM traffic; no
throughput or overlap advantage is claimed.

`make_final_coverage_agreement` aggregates local reasons and unconditionally
invokes the existing S5 request MAX collective, returning common error plus
local summary. Every rank must call, including empty ranks. This supplies a
coverage decision only: transport errors, history consistency and completed
DMA/consumer drains still must gate final publication and scratch reuse.
The function neither publishes frontier nor establishes those other drains.

Local coverage tests: seven passed53.98s, including 129 live records scattered
across capacity256, paddingUINT32_MAX, duplicate/missing/extra/overflow and
empty inputs (`local_final_target_coverage.xml`). Single-rank agreement and
eight-rank tracing are separate tests. Physical eight-TPU compilation and
collective execution of this composition remain pending. No full beam proof.

Full unchanged-snapshot regression:907passed1522.85s, zero failures/errors/skips,
with both source C++ oracle paths enabled (`local_final_coverage_full.xml`).
This is local regression evidence only. Agreement focused tests:3passed13.82s.
