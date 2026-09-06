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

## Accumulated transport/history errors

The pending extension accepts `prior_error: uint32[1,128]`, with an accumulated
local error in lane `[0,0]`. The common flag is the normalized logical OR of
coverage failure and this prior error, before the unconditional collective.
A nonzero high bit is an error, not a signed negative value to be lost by MAX.
The returned local summary deliberately remains coverage-only: callers retain
their transport/history diagnostics separately.

Focused evidence: `local_final_prior_error.xml` has four passing tests,
including valid coverage with prior flag `0x80000000`. The expanded physical
fixture/coordinator tests have eight passes (`local_final_prior_bundle.xml`).
The physical bundle now requires 6 CUDA-byte cases, 16 exchange cases and
7 coverage cases. The new coverage case places a prior error on rank6 with
valid target coverage everywhere; all ranks must reject publication.

Full regression `local_final_prior_full.xml` has completed:909passed1166.12s,
zero failures/errors/skips, with both C++ oracle paths enabled. These changes
are locally verified, not physically TPU-accepted. The prepared final launcher
must pin this verified public source before submission. S5 V9 is now accepted
separately and no longer occupies a session.
Neither a zero common flag nor exact coverage proves that DMA/history consumers
have drained; that synchronization remains a separate caller obligation.
