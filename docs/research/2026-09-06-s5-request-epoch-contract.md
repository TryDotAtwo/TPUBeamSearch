# S5 request epoch: source contract and pending implementation

Read-only `D:/100XH100/cuda/dispatcher.cu`, SHA256
`45ccfe9ddd27886acbb90ebb30ab67ef96a0557e2c7577043ac9aa3fc1b6027b`.
Inspected `maybe_run_stream5_threshold_update` around3001 and the unconditional
boundary call around3744. This is an implementation contract, not acceptance.

1. Each rank computes request = forced OR periodic_due. Every rank enters the
   same request collective, including ranks with no completed local S4 jobs.
2. Reduce requests with uint32 MAX. Only the common zero result skips histogram
   work. A rank's own zero request is never permission to skip the collective.
3. A common nonzero result triggers local committed histogram snapshot, global
   uint64 SUM and periodic threshold selection/publication on every rank.
4. Complete the threshold update before incrementing update count, resetting
   jobs-since-update, and clearing local/global requests. No-request epochs do
   not reset the accumulated job count.
5. At the boundary before final drain, every rank calls the request collective;
   local outstanding jobs force a request, but absence of them cannot skip it.

The source uses host synchronization for the decision; the TPU implementation
must preserve the decision and order without claiming that asynchronous Python
dispatch supplies a protocol. An initial serialized epoch is valid; overlap
requires separate ownership and a physical profile.

## Required TPU acceptance

- Requests from each individual rank; all-zero and all-one requests; empty ranks.
- Two or more consecutive epochs with no stale semaphore/request state.
- Exactly one common decision; no rank-local branch around remote operations.
- Histogram uint32-pair accumulation preserves carry through global reduction.
- Frozen selected histogram versions remain valid until all snapshot reads end.
- Threshold slots cannot be reused while captured readers still depend on them.
- Reset job/request counters only after successful common publication.
- Include the unconditional pre-final boundary epoch in multi-depth replay.

Current local snapshot, threshold arithmetic and DMA publisher cover pieces of
step3 only. Request reduction, rank coordination, snapshot reader ownership and
counter lifecycle are still missing; a passing local publisher is not S5.

## Request primitive under test

`make_s5_request_call` now sends each immutable original request to all peers
in serialized peer-offset rounds. Destinations use separate slots per offset;
each transfer waits for send and receive, and every rank participates even for
zero requests. A second Pallas call computes MAX. This is deliberately not an
overlap or scalable collective performance claim.

Initial missing-module test failed; two single-rank interpreter cases and an
eight-rank JAXPR ABI test pass (3 tests,3.40 s). Neither proves distributed DMA
execution. Physical acceptance must exercise all-zero, each singleton requester,
all-one, and repeated epochs on the same executable. No current Kaggle session
has been changed to include this code. Full local session48457 was collected
before this module existed and therefore does not cover these three tests.

`benchmarks/beam_s5_request_probe.py` now prepares20 sequential cases on one
compiled eight-device executable: zero, each singleton requester, all ranks,
then the entire sequence again. It saves partial per-case hashes/mismatches,
runtime/source/device inventory and HLO, and rejects non-TPU execution. No
performance ratio is inferred from these correctness calls. A fixture coverage
test was red before implementation; the combined request/probe checks pass4
tests in3.61 s (`test_results/local_s5_request_regression.xml`). The physical
probe has not been submitted and is not part of the queued collector V3.

`pallas_sum_histogram_pairs` adds the arithmetic needed after receiving rank
histograms: it adds each high word as well as low-word carry. A red-before-code
test then passes in4.89 s against NumPy uint64, with eight ranks/two tiles,
random values below2^59, eight UINT32_MAX entries and eight0x100000001 entries.
This is local arithmetic only; global histogram transport is not connected.
The sum must fit uint64 and input padding must be zero by contract. This test
was also added after full-regression48457 collection.

`make_s5_histogram_call` now connects peer-offset histogram all-gather to that
pair sum. The local input remains immutable; each peer offset has a distinct
HBM destination. All sends/receives are waited. It uses2*ranks*width uint32
scratch:19,668,992 bytes per rank for eight ranks and padded width307328.
Input, reduced output and other persistent buffers are additional. This is
not a reduce-scatter or overlap claim. Single-rank two-tile race interpretation,
eight-rank JAXPR ABI and carry arithmetic pass3 tests in6.10 s; artifact:
`test_results/local_s5_histogram_regression.xml`. Physical distributed execution
is still pending. The full local regression was collected before this module.

The same physical driver now accepts `--kind histogram` and writes
`s5_histogram.json`/HLO separately. Six calls on the same executable cover zero,
mixed uint64-pair contributions, only rank5 nonempty, then repeat including a
zero-reset case. Expected global sums use bounded NumPy uint64 and are compared
on every rank. The new fixture test was red first. All8 request, histogram,
pair-sum and fixture checks pass in5.77 s:
`test_results/local_s5_combined_regression.xml`. These remain local tests; neither
mode has been submitted. Future physical runs must isolate modes in separate
processes to retain diagnostics if one compiler invocation aborts.

## Publication checkpoint

Full local regression48457 is terminal:732 passed in1064.69 s, no failures,
errors or skips. It includes the periodic publisher, but not the later S5
modules. A combined supplement then passed21 tests in6.24 s, covering that
publisher, the new request/histogram primitives and fixtures, and an isolated
coordinator (`test_results/local_s4_s5_supplement.xml`). These counts overlap;
do not sum them as distinct tests.

Prepared `benchmarks.beam_s4_s5_bundle` runs reserved S4, request MAX and
histogram SUM in separate sequential processes. It retains partial JSON/logs
and continues independent groups after native abort; returncode must be zero
even if a partial nested report says exact. Its missing-module test was red
before implementation. No new TPU session has been submitted; collector V3
remains queued. This bundle is not yet an integrated S5 lifecycle benchmark.
