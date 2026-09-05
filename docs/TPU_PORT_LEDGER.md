# Whole-architecture TPU port acceptance ledger

The user's completion condition is the whole architecture plus a GPU/TPU
comparison, not isolated kernels. No complete-port or performance claim yet.

Collector V2 is now terminal ERROR, not pending. Single and grouped append are
exact on eight TPU v5 lite devices; full collector fails compilation at dynamic
VMEM offsets[0,shard_id] (E2003 alignment). Complete outputs are retained in
`test_results/beam_collector_v2/`; see report.md for scoped timings and hashes.
A failing structural regression catches the exact scalar load; aligned vector
selection/reduction fixes it locally. Six gather/scatter tests pass in8.81 s.
Physical full-collector confirmation is still required before the integrated
S3/RDMA/collector gate. The earlier full run completed704 tests with zero
failures/errors/skips in1227.68 s; it predates this correction. The corrected
full run completed712 tests in1033.87 s with zero failures/errors/skips:
`test_results/local_collector_v3_regression.xml`.
Collector V3 is terminal ERROR (source12aae5b085a58ff81eec60ac1eb73009cae927c0,
launcher7c54e20fd1c68135a888ec0dcf1d54f3accdd701). Its recovery coordinator runs
full first and integrated S3 only after exact success. Outputs are downloaded
into `test_results/beam_collector_v3/`: full compile rejects8x256->8x128 gather
across multiple source vregs; integrated was not started, no timings exist.
Candidate128-column banked gather passes7 local tests; full regression82378
completed764 tests in1066.95 s with zero failures/errors/skips:
`test_results/local_collector_v4_regression.xml`. Physical V4 remains pending.

The next standalone physical S4 gate is prepared in `benchmarks/beam_s4_probe.py`,
not submitted. It checks a reserved128 job across eight devices, including both
histogram slots and empty/nonempty outputs, before diagnostic3/21 timing. Four
fixture/operator interpreter tests pass in59.79 s (missing-module tests were
red first); XML: `test_results/local_s4_probe_regression.xml`. It intentionally
does not claim ready-queue integration, production307201-bin scalability,
global S5, concurrent snapshot safety or beam performance.

Post-da90c64 local work (not yet physically accepted): external S4 dedup retains
score/parent64/route winners without a shard cap; CPU source comparisons pass.
Sorted-score histogram and explicit DMA publication primitives pass interpreter
checks. One reserved physical S4 job composes clean+dirty count, threshold,
dedup, histogram and aliased record/histogram/control commit. Two sequential
jobs verify empty-result clearing and histogram flip-back. It still needs
logical A/B scheduler integration, bounded production scratch, full source
replay and compiled alias/memory/profile evidence. Local committed-histogram
snapshot returns uint32 low/high pairs and passes a carry test; concurrent
snapshot ownership and coordinated S5 request/reduce epochs are not implemented.
Details and individual test scopes are in
`docs/research/2026-09-05-resident-s4-publication-contract.md`.

The bounded128 S3/snapshot-exchange/collector composition now has a saved
eight-rank source-backed fixture and a physical TPU gate harness. Local replay
checks every A/B/control/fatal element with a simulated network and passes
(with its ABI test: 2 passed in 353.12 s). Remote snapshots are assembled in
ascending source order and admitted as one batch, separately from local input,
matching the audited CUDA dispatcher boundary. Functional copies, retained
snapshots, coordinated stop and physical composition remain open. This is not
CUDA execution, real RDMA, overlap or full-beam performance evidence. See
`docs/research/2026-09-05-collector-transport-integration.md`.

Full composition regression: 675 passed in 921.58 s, with source and route CPU
oracles enabled; `test_results/local_stream3_collector_regression.xml`.
The later S4 ready selector has 21 separate passing tests in 8.95 s, recorded
in `test_results/local_s4_ready_regression.xml`; it was not collected in that
full run. See the resident S4 publication audit for remaining residency,
histogram completion, queue integration and physical validation requirements.

Collector V1 terminated with a Mosaic gather lowering rejection on eight TPU
v5 lite devices: `Only take_along_axis-like gathers supported`. No device
correctness or timing was produced; see `test_results/beam_collector_v1/report.md`.
The concrete JAXPR index-shape mismatch is reproduced for append/scatter and
fixed with broadcast indices plus take_along_axis. Fifteen related local tests
pass. The next sequential single/group/full collector bundle is prepared.
Final regression including scatter, gather correction and harness: 664 passed
in 435.19 s, zero skips/errors/failures, with source CPU oracle enabled; XML
`test_results/local_collector_bundle_regression.xml`. Private collector V2
submitted and confirmed QUEUED: source `bad92c169a1001878ccb625f609c6cb634585b53`,
launcher `7743d33`. It runs isolated single/group/full checks sequentially.
Download terminal output to `test_results/beam_collector_v2/`; inspect bundle
and every nested JSON/log. Pending is not physical acceptance.

Hash-to-shard collector grouping now has an interpreter baseline: stable
metadata partition, counts/offsets, and actual independent Hash128 shard salt.
Three supplied-ID cases and one independent uint64 hash-composition case pass.
Physical compilation and resident scatter/publication remain unverified.
Full regression: 654 passed in 466.33 s, no skips, with original source CPU
oracle enabled; XML in `test_results/local_collector_routing_regression.xml`.
The subsequently added all-shard preflight passed its two separate interpreter
tests. See the collector/transport integration note for CUDA batch-fatal scope.

Variable-count snapshot transport has been extracted from the diagnostic
benchmark into `beam_remote_exchange.py`. The function body is unchanged
(at extraction, compared directly with the prior Git version); the benchmark imports it.
Eleven related local tests pass (1.70 s), including a new traced output-ABI
test that was red before extraction. This is not fresh physical execution.
Nonzero copies still transfer full capacity and keep one snapshot per epoch;
neither direct collector integration nor count-proportional traffic is proved.

The next full suite reported 651 passed / 1 failed: RDMA `lax.rem(epoch,2)`
mixed an int32 program ID with an int64 Python literal when x64 was enabled.
A two-mode ABI trace reproduced that failure only with x64 enabled. Explicit
int32 remainder constants fix it without changing the DMA ordering; 12 related
tests now pass. The full 654-test regression above also passes after this fix.

Multi-tile collector extension is under local validation:
`pallas_collector_append_group` preflights the entire group against one sibling
before any per-tile append, so insufficient whole-group capacity cannot produce
a partial write or distribute the group across siblings. Three red tests then
passed (10.09 s), including 256/257 records and exact-full. All sixteen related
collector/adapter tests pass (17.27 s). This still uses functional buffer copies
and a whole-tuple completion boundary, not resident aliasing or concurrent
count publication. Physical collector V1 remains pinned to the prior single-tile
source; this extension is not included in that pending result.
Full local regression for this extension: 647 passed in 412.75 s, with the
original CPU C++ oracle enabled. Physical multi-tile confirmation remains open.

Serialized collector work in progress: `beam_collector.py` now has a Pallas
functional append for one A/B pair and a group of at most 128 records. Five
initial interpreter tests passed (6.63 s); physical lowering is unverified.
It preserves other records, uses clean+dirty offset, prefers writable current
then sibling, and latches fatal instead of dropping a group. This is NOT the
final resident collector: it returns whole buffers, has no proven donation,
and requires waiting for the entire tuple before publishing control. The
control call has no record-store dependency on its own; a concurrent consumer
must not observe it independently. Production integration needs explicit
commit ordering, grouped multi-tile reservations, shard selection and DMA
lifetime checks. No overlap or allocation-efficiency claim is made.
Full local run: 642 passed in 540.06 s. Additional repeated-arrival/sticky-fatal
coverage passes with all six collector tests (21.93 s), and the later physical
harness adapter test passes (4.88 s). Those additions were checked separately
from the already running full suite. No physical collector result yet.

Composed external S3 V1 is physically exact at N256/512 on eight TPU v5 lite,
source `bb6c38a`, launcher `9619cdc`; see
`test_results/beam_external_stream3_v1/report.md`. Threshold/dedup/actual owner/
split match recorded original CPU C++ fixture outputs for all five arrays.
Diagnostic medians 0.77270/0.83023 ms are not beam speedups. Source fixture
checkout was dirty, with actual file hashes retained; CPU parity is not CUDA.
Collector/RDMA and ring payload restoration remain outside this gate.

Composed external S3 now executes threshold/dedup -> actual Hash128 owner ->
external split in `pallas_external_stream3`. An original C++ Stream3 oracle
fixture (N256, world8, rank3, threshold5, reverse payload tie priority and
nonzero parent high words) was red before implementation, then passed in
35.11 s. It checks local/remote metadata, counts, offsets and neutral tails.
Full local regression: 629 passed in 428.54 s with CPU C++ oracle enabled.
This is interpreter/source parity, not physical composed TPU or CUDA evidence.
Metadata must already correspond to payload IDs; ring payload restoration,
collector and transport are not implemented by this composition wrapper.

External S3 split V1 is physically exact at N256/512/1024 on eight TPU v5 lite;
see `test_results/beam_external_split_v1/report.md`. All five output arrays
match independent partition expectations and hashes. Diagnostic medians are
0.73925/0.75874/0.82799 ms. Supplied-owner split and its whole count scratch
are validated at these shapes only; real owner-after-dedup composition and
collector/RDMA integration remain pending. Source `c73f601`, launcher `fc6a7f5`.

External Stream3 dedup V3 is physically exact at N=256 and N=512 on all eight
TPU v5 lite devices, source `5e08ff60f3d470cef6ccdf0fc173510a827aecd3`.
See `test_results/beam_external_dedup_v3/report.md`: zero metadata/count
mismatches and equal output hashes, diagnostic medians 0.72310/0.76190 ms.
This closes the external dedup scratch-layout gate only. Routing after dedup,
collector integration, larger-N efficiency and whole GPU/TPU replay remain open.

External S3 split now has a diagnostic Pallas implementation in
`beam_external_sort.py`, using tiled external sorts instead of whole-N gather.
It consumes supplied owners and aligned `[1,128]` counts, preserving stable
local/owner-grouped remote order, route fields, parent high words and neutral
tails. Four N256 tests were observed red before implementation and then passed
in 64.17 s. Full local suite including original CPU C++ oracle: 626 passed in
422.43 s. This is not physical split evidence or composed source-owner parity.
The `(world_size+1,N)` count scratch still uses a whole-window reduction and
must be checked for physical lowering/VMEM limits; it is not an efficiency claim.
After the full run, coverage was extended to mixed N512: all five split tests
passed in 75.84 s. No production code changed after that full run.

| Contract | Implementation/evidence | Remaining acceptance |
|---|---|---|
| Logical types / uint32 SoA | beam_types.py; high-word and padding tests | all consumers, physical TPU |
| Local pipelined transport | beam_transport.py, 2/3 input buffers; race simulation | physical DMA trace and throughput |
| S1 MOVE_COUNT score producer | existing inference engines only | exact selected-model integration, quantization without HBM float-Q |
| S2 immediate hash / exact goal | beam_stream2.py; independent C++ source oracle | real TPU compile, valid-input launcher, K1/K2, bounded solved collection |
| Hash128 owner/shard arithmetic | beam_hash.py uint32 pairs; edge/random modulo tests and C++ oracle | physical TPU lowering, fingerprint and K1 bucket salts |
| S3 threshold/sort/dedup | beam_dedup.py plus beam_stream3.py bounded Pallas split; source differential tests | physical split compile, HBM-scale sort and collector integration |
| S4 threshold/sort/dedup | same primitive with score/parent64/route tie-break; C++ oracle | resident A/B collector, independent capacity/trigger, committed histogram |
| S5 | beam_dma_ring.py host ordering oracle; readiness/send/recv/consume/ack, zero-count and wrap tests | Pallas remote DMA, coordinated threshold, race and physical TPU tests |
| Three scratch overlays | design only | explicit arena plan, alias report, drain gates |
| Final | source read, design only | exact global cap/ties, balance, request/response, padding, history, replay |
| Stop | design only | bounded solved records, no collective deadlock, multi-rank stop |
| Whole depth / multi-depth | not implemented | original CUDA replay and 8-TPU replay on identical fixtures |

`beam_dma_ring.py` now makes the S5 slot-lifetime contract executable before
introducing remote DMA: destination readiness precedes a nonzero start, source
reuse follows send completion, destination reuse follows receive, consumption
and acknowledgement, and a zero-count rank participates without waiting for a
DMA that was never started. Four tests cover two-slot reuse across four epochs.

The first physical Pallas RDMA gate is exact on eight TPU v5 lite devices; see
`test_results/beam_rdma_ring_v1/report.md`. A 32 KiB/device right-neighbor push
with distinct DMA send/receive semaphores and explicit start/send-wait/recv-wait
matched the independent rotation oracle with zero mismatches. Median diagnostic
call time was 0.521 ms, but this is neither a variable-count S5 exchange nor
overlap proof. Readiness, two-slot ack/reuse and zero-count physical gates remain.

The two-slot physical gate now passes on eight TPU v5 lite devices; see
`test_results/beam_rdma_ring_v2/report.md`. Four epochs exercise one wrap of
both HBM destination slots with readiness, separate per-slot DMA semaphores,
receiver consumption and acknowledgement before reuse. Both all-active and
alternating zero-count cases are exact with zero mismatches; zero-count epochs
start and wait on no remote DMA. Next is a real per-edge variable-count Stream3
exchange with explicit capacity and count metadata, not a synthetic fixed-size
ring.

Variable-count RDMA V3 failed before execution; see
`test_results/beam_rdma_ring_v3/report.md`. Root cause is direct predicate
loads from `pl.ANY` count references, which Mosaic correctly rejects because
HBM must be accessed through async copies. The local send count is now scalar
prefetched and the received count is staged into per-slot VMEM before branching.
V4 physical confirmation is required; no Stream3 correctness claim is made yet.

Variable-count RDMA V4 compiled but the benchmark then failed in its shared
warmup loop; see `test_results/beam_rdma_ring_v4/report.md`. Its four-input
compiled executable was called with one tuple argument. Invocation is now
centralized so tuple placements are splatted while legacy single-input probes
retain their existing call shape. V5 still requires physical correctness and
timing confirmation.

Variable-count RDMA V5 is physically exact on eight TPU v5 lite devices; see
`test_results/beam_rdma_ring_v5/report.md`. Across all seven peer offsets it
matches independent payload and count hashes with zero mismatches, including
zero-count epochs, a capacity-128 boundary, neutral fixed-slot tails and
two-slot reuse. The isolated diagnostic median is 0.6545 ms over 21 samples
after three warmups. This validates transport only; the next gate is one
compiled `pallas_stream3_split` output-layout to variable-exchange boundary.

Integrated split-to-RDMA V6 failed during Mosaic lowering; see
`test_results/beam_rdma_ring_v6/report.md`. The wire adapter used advanced
integer indexing directly on a Pallas Ref, which TPU rejects. It now loads the
whole aligned `[8,128]` block before permuting the local value. No integrated
correctness claim is made until V7 physically compiles and matches the oracle.

Integrated V7 reached the next lowering boundary: array advanced indexing
became a general gather, while Mosaic supports only take-along-axis-shaped
gathers for this layout. See `test_results/beam_rdma_ring_v7/report.md`. The
wire adapter now broadcasts the index to `[8,128]` and explicitly uses
`take_along_axis(axis=1)`. V8 physical confirmation remains required.

Integrated V8 accepted the payload gather and then failed on the dynamic
`count_ref[0, peer]` tiled-VMEM scalar load with Mosaic E2003 unproven
128-lane alignment; see `test_results/beam_rdma_ring_v8/report.md`. Count and
offset control vectors are now loaded as aligned blocks and device-dynamic
peer values are selected by a one-hot mask reduction. V9 physical confirmation
is required.

Integrated V9 passed aligned control selection but Mosaic rejected reduction
over `uint32`; see `test_results/beam_rdma_ring_v9/report.md`. The bounded
count/offset masks now reduce as `int32` and cast back to `uint32`, matching the
already compiled Stream3 count pattern. V10 physical confirmation is required.

Integrated V10 passed control reductions but Mosaic could not legalize the
unsigned vector minimum used to clamp gather indices; see
`test_results/beam_rdma_ring_v10/report.md`. The bounded index arithmetic and
minimum now run as `int32`, without changing selected lanes. V11 physical
confirmation is required.

Integrated V11 is physically exact on eight TPU v5 lite devices; see
`test_results/beam_rdma_ring_v11/report.md`. One compiled program executes
bounded N=128 Stream3 split, device-rank route formation, ring-ordered wire
packing and seven-epoch variable RDMA. Local, wire and received payload/count
outputs match the independent oracle with zero mismatches, including a
capacity-128 edge, zero edges and neutral tails. Median diagnostic latency is
0.80834 ms over 21 samples after three warmups. This closes bounded
split-plus-transport correctness only; HBM-scale Stream3 remains open.

This is a host race oracle, not a Pallas implementation or TPU evidence.

## Evidence boundaries

External dedup V2 passed shape validation, then TPU lowering rejected the
count output block (1,128) over (2,128). See
`test_results/beam_external_dedup_v2/report.md`. Tile count scratch is now
[1,tiles*128] with aligned column blocks; external count/storage semantics
remain unchanged. V3 physical acceptance is required; V2 has no timings.

External dedup V1 failed before TPU lowering: the benchmark passed [1,1]
count/threshold controls to the primitive's [1] contract. See
`test_results/beam_external_dedup_v1/report.md`. A failing per-shard eval_shape
regression reproduces the error; the adapter now removes both singleton axes.
Production Pallas is unchanged; physical V2 acceptance remains open.

External S3 threshold/dedup now has a tiled Pallas candidate implementation in
`beam_external_sort.py`: inclusive threshold, Hash128/score/payload sorting,
predecessor comparison across tiles, stable compaction and neutral/count output.
Routing remains after dedup. It retains parent high words and applies no cap.
Eight focused interpreter tests pass; physical TPU acceptance is pending.
Full regression: 614 passed, 5 skipped in 360.46s; the five optional original
C++ checks passed separately with BEAM_SOURCE_ORACLE set. A subsequently added
256-record original Stream3 C++ comparison also passed (four new-file tests).
The compiled C++ adapter is a CPU oracle, not CUDA execution evidence.
The temporary N<=16384 bound limits count scratch; 36/45-pass sorts at N256/512
and a second compaction sort are diagnostic, not the final scalable algorithm.

External HBM sort V1 is physically exact on eight TPU v5 lite devices; see
`test_results/beam_external_sort_v1/report.md`. N=256 with 128-column tiles
matches all 11 uint32 planes and output SHA256. Median diagnostic latency is
0.48990 ms (3 warmups, 21 samples; p10/p90 0.46301/0.53411 ms). This is a
36-pass bitonic correctness baseline, not an efficient large-N implementation
or a beam speedup. The JSON `runs=2` means tiles, not independent correctness
trials. Threshold/cross-tile dedup/compaction/count and larger-N gates remain.

2026-09-03 local verification: 535 tests passed in 158.84 s with
BEAM_SOURCE_ORACLE enabled, including both original C++ differential tests.
No skips in this run. JAX 0.10.1 CPU interpretation; no physical TPU or CUDA
execution evidence for these new primitives yet.

The adapter `tests/beam_source_oracle.cpp` links original `src/hash.cpp`,
`src/state.cpp`, `src/stream4.cpp`, `src/stream3.cpp` from D:/100XH100 read-only. It executes on the
CPU. It is stronger than a reimplemented Python oracle, but not CUDA execution.
Run from repo root:

```powershell
cmd /c tests\build_beam_source_oracle.cmd
$env:BEAM_SOURCE_ORACLE = (Resolve-Path .local/beam_source_oracle.exe).Path
python -m pytest tests/test_beam_source_parity.py -q
```

GPU source identities inspected 2026-09-03 (SHA256):

- src/hash.hpp: 361756fe2de60ae9393f0e60f6be80c697e9b84c58fbaedbc75d1c5d8162016c
- src/hash.cpp: 369b34d3eaa526b002b553ee4bf2e9ee1eccd69861c6cac7f1d06a5162b65136
- src/state.cpp: 7094caed55c9876eb496a4e5c705b814309b5d3807b7e935712b78d09614406c
- src/stream4.cpp: d233acefb23c030057de8a92f6de1fda55617ef925d757a49d7742b736bb72f2
- src/stream3.cpp: 584d37e0c96d581174359932a2fe04df76cbfdfca12b44b0a5df75c3aec1d2b5
- cuda/stream2.cu: d52252daba39fc913a31c7ded25f08721b8b58aa8c5ca0e09872bf7a6e61f126

## Next execution order

V4 COMPLETE, all_exact=false. See `test_results/beam_primitives_v4/report.md`.
Five exact cases. Isolated hash120 abort no longer blocks hash150/dedup. Hash150
identifies unsupported uint8 flatten; dedup identifies unsupported uint32 sum.
The count <=4096 permits int32 sum with uint32 output unchanged. Hash kernels
are not modified speculatively. New split uses separate local/remote buffers,
stable owner grouping and padded uint32 control planes, but is bounded bitonic
diagnostic code, not an HBM-scale partition or remote DMA implementation.
Final local verification after the split and signed-count changes: 556 passed
in 280.73 s, original C++ oracle enabled. Two split cases join the next physical
bundle (13 total cases); local tests do not establish TPU lowering success.
V5 source b469a6863a6ca3472c71022245d7d3ef4f86be65. Download into
`test_results/beam_primitives_v5/`; inspect all ten groups / thirteen cases.
Unchanged hash errors remain expected unresolved failures, not accepted cases.

V5 COMPLETE, all_exact=false; report: `test_results/beam_primitives_v5/report.md`.
Five exact controls. Signed count reduction moved dedup to its next compiler
boundary: scalar VMEM store; both split cases expose the same issue. V6 stores
logical counts in lane [0,0] of padded [1,128] uint32 control planes, with all
other lanes zero. Structural tests reject scalar stores; interpreter and source
oracle parity remain exact. Physical confirmation is outstanding.
Full local verification: 559 passed in 220.01 s with original C++ oracle enabled.
V6 source 9f2d083aa51bfe240ebaae3243d0fff20112dbc9; new output directory
`test_results/beam_primitives_v6/`. Same thirteen cases isolate physical effect.

V6 COMPLETE, all_exact=false; report: `test_results/beam_primitives_v6/report.md`.
The scalar stores are gone. Six dedup/split cases now share Invalid input layout
at select_n, specifically i1->i8 vector extension. V7 removes only control-plane
boolean selects, using uint32 indicator multiply/add while leaving data sorting
unchanged. Structural regression was red for all three representative cases and
is green after the change; physical confirmation remains required.
Full local verification after the V7 change: 562 passed in 290.50 s with the
original C++ oracle enabled.
V7 source 2eb6aa7b40c9b56456c4bfc1627504904b4dc851; download into
`test_results/beam_primitives_v7/`. Same thirteen cases isolate the select change.

V7 COMPLETE, all_exact=false; report: `test_results/beam_primitives_v7/report.md`.
Five controls remain exact and both hash failures are unchanged. All four dedup
and both split cases retain the identical V6 `select_n` i1-to-i8 invalid-layout
compile failure. The control-plane hypothesis is therefore falsified: the next
step is a minimal physical selector probe for the shared survivor data path,
not another broad production rewrite.

Selector probe V1 COMPLETE; report:
`test_results/beam_selector_probe_v1/report.md`. All five plain `[8,128]`
selector forms are physically exact, including broadcast/full boolean `where`
and boolean/uint32 arithmetic masks. Thus no isolated selector form reproduces
V7: the rejected layout is created by interaction with an earlier full-pipeline
operation. The next physical probe bisects the actual dedup stages before any
production change.

Dedup stage probe V2 COMPLETE; report:
`test_results/beam_selector_probe_v2/report.md`. Initial 11-plane construction
is exact; the first bitonic sort is the first invalid-layout boundary. Thus the
failure precedes uniqueness, second sort, final selection and count storage.
The next probe isolates one real compare/exchange (partner gather, predicate,
and alternative gathered-data selection) before changing production `_sort`.

Compare/exchange probe V3 COMPLETE; report:
`test_results/beam_selector_probe_v3/report.md`. Partner gather is exact, while
emitting the swap predicate alone reproduces the invalid-layout failure. Thus
the root expression is the conditional `where(want_min, ~less & ~equal, less)`;
the downstream data selector is not reached. V4 tests its equivalent pure
boolean formula and one complete compare/exchange before production changes.

Compare/exchange probe V4 COMPLETE; report:
`test_results/beam_selector_probe_v4/report.md`. The pure boolean swap identity
and a complete gather/compare/select using it are physically exact, while the
conditional boolean `where` remains rejected. Production `_sort` now uses the
proven identity. The new regression was red on dedup and both split cases and
green after the one-expression change. Targeted source parity passes; full
local verification: 581 passed, 5 skipped in 264.60 s. The next full physical
primitive bundle must confirm all four dedup and both split cases.

V5 previous-index probe completed on all eight TPU v5 lite devices; see
`test_results/beam_selector_probe_v5/report.md`. At the full
`sort -> previous hash -> uniqueness` boundary for `N=128`, the existing
`maximum(indices,1)-1` reproduces the `arith.maxui` compile failure, while both
`where` and branchless subtraction are physically exact with zero mismatches.
Production now uses `indices - (indices != 0).astype(uint32)`, with a structural
regression that was observed failing before the change. This removes only the
V8 `N=128` blocker; the independent `N=256` multi-source-vreg gather remains.

V9 full primitive gate compiled and executed both `N=128` dedup modes, proving
the `arith.maxui` blocker is removed; see
`test_results/beam_primitives_v9/report.md`. Both were labelled structural
mismatches because the benchmark oracle retained the pre-alignment count shape
`[1]`, while the production ABI and TPU output are `[1,128]` with lane-zero
semantics. A new regression reproduced this host-oracle defect and the oracle
now uses the aligned plane. V10 must confirm value-level exactness. `N=256`
and both hash compiler failures are unchanged and remain separate workstreams.

V10 closes the bounded `N=128` dedup gate on physical eight-device TPU; see
`test_results/beam_primitives_v10/report.md`. Stream3 and Stream4 compile,
execute and match their independent aligned oracles with zero element
mismatches, including the `[1,128]` count plane. Diagnostic medians are
0.661 and 0.672 ms respectively, but are cross-process measurements and not a
matched A/B or complete beam claim. `N=256` still needs an HBM-scale staged
sort/merge that avoids a two-vreg gather; the two hash lowering failures remain
independent.

Primitive gate V8 COMPLETE, all_exact=false; report:
`test_results/beam_primitives_v8/report.md`. Both Stream3 split cases are now
physically exact on eight TPU, proving the swap-predicate fix through complete
split execution. Dedup also passes that boundary: N=128 next fails at unsigned
`maximum(indices,1)` for previous-hash addressing, while N=256 exposes a
separate multi-vreg gather limitation. Hash failures remain unchanged. Split
success is bounded local partition evidence, not HBM-scale or remote DMA.

V5 COMPLETE, all_exact=false; see `test_results/beam_primitives_v5/report.md`.
The signed reduction passed its prior lowering boundary, then all dedup and
split cases reached `Cannot store scalars to VMEM`. V6 represents logical counts
as aligned `[1,128]` uint32 control planes and writes whole vectors. This is a
storage-layout change only: logical value remains `[0,0]`, unused lanes zero.

Stream3 source adapter added while V4 remains QUEUED. All five source-oracle
tests pass (18.57 s), including three new Stream3 cases: world=8 with payload
ties opposed to parent order, world=1 with UINT32_MAX threshold, and empty input.
Pallas CPU interpretation reproduces survivor metadata; host test partition
matches original C++ local/remote ordering, counts and offsets. This does not
implement or validate TPU partition/DMA. No production primitive changed.
See `test_results/beam_stream3_source_parity.md` for the scope and reproduction.

V10 prefix session completed; see
`test_results/artgor_reduction_geometry_v10/report.md`. Transposed JAX reduction
on fixed Dense/mean reproduces original prefix exactly, normal geometry does not.
This is a useful S1 clue, not a completed Pallas inference path. No automatic
standalone inference sweep is restarted.

Physical primitive gate completed as private Kaggle V1:
`trydotatwo/tpu-beam-primitive-compile-and-correctness-gate`, source
`c05b269a09edc38c846840d2fb433848b83a7986`, launcher `f24db82`.
Kaggle assigned the title-derived slug on first creation; metadata was then
corrected to the actual slug without resubmitting. Download into
`test_results/beam_primitives_v1/` and inspect every case, even if kernel status
is COMPLETE (compile errors are retained per case and do not abort the bundle).
V1 result: five exact cases (packing and routing), six gather compile rejections;
no timings collected. See `test_results/beam_primitives_v1/report.md`. Gather
fixes have structural regression and unchanged CPU/source parity; physical
confirmation is required in V2, which also adds eligible-case timing.
V2 source: ef7512627d1ba9c58c5c444391107d1887d2ee84. Local verification:
540 tests passed in 146.41 s with original-source C++ oracle enabled. Output
destination for the new run is `test_results/beam_primitives_v2/`. Pack timing
uses 65536 candidates/device, three warmups and 21 alternating synchronized
samples; only exact cases are eligible. Completed results follow below.

V2 is now COMPLETE; report: `test_results/beam_primitives_v2/report.md`.
Five exact cases; six compile failures moved past gather to scalar uint8 extract
(Stream2) and sentinel scatter (dedup). Packing medians serial/pipeline:
b2 1.347320/0.670960 ms (2.008x), b3 1.335930/0.632890 ms (2.111x), at 65536
candidates/device. These are primitive-call measurements, not beam/inference
speed or profiler-confirmed overlap. V3 fixes passed 542 local tests in 159.72 s,
including original-source C++ parity and both new failing-before-fix regressions.
V3 source e9c90a9ef3f132fce651218f20b62707d741e21e; download new outputs into
`test_results/beam_primitives_v3/`. Physical confirmation remains outstanding.

V3 ended ERROR: five exact cases, native VectorLayout::join compiler abort at
hash_goal_120_24, remaining five cases not attempted; no timings. See
`test_results/beam_primitives_v3/report.md`. The next bundle isolates each
non-packing case in a sequential process, retaining partial JSON and full logs.
The failing source expression is not yet identified; primitives stay unchanged.

V4 source ea3eaa09822777e47c65101299fac7093bb8d952 uses the isolated coordinator.
Local full suite: 544 passed in 157.05 s with source oracle enabled; the expanded
isolation file separately passed all three tests. New output directory:
`test_results/beam_primitives_v4/`. Inspect `beam_primitives/isolated_bundle.json`
and each group's nested report/log; native process failures may coexist with a
COMPLETE Kaggle status. Cross-process results are not matched latency ratios.

1. Validate these primitives on physical TPU before propagating their lowering
   choices. Preserve any QUEUED/RUNNING current gate; never run a second TPU job.
2. Add a standalone remote-DMA ring test: readiness, send/receive wait, slot
   acknowledgement, zero-count and multiple-wrap cases, serialized control.
3. Validate Pallas Stream3 grouping physically; expand source oracle to final request/response. Build
   HBM-scale sort/merge and preallocated A/B collector with fatal overflow.
4. Integrate K1/K2, histogram epochs, final phases and stop, one tested subsystem
   at a time. Keep every row above explicit until its acceptance evidence exists.
5. Compare discrete search with identical score tensors first. Then compare actual
   inference separately; FP16/BF16 differences cannot excuse a search logic defect.
6. Compare whole GPU/TPU depths, selected identities, hashes, parents/routes,
   history replay, overflows, ties, valid counts and solutions; separately measure
   runtime and overlap. Record any allowed layout/scheduling changes explicitly.

### Stream2 K1 composition — local only, 2026-09-06

`beam_stream2_k1.pallas_hash_k1_goal` preserves immediate-child hashes and
validity, then replaces exact-central flags with full-Hash128 K1 membership.
Parent-to-child count conversion is a Pallas dispatch. The existing Stream2
default remains unchanged. Caller supplies an enabled K1 table containing the
central state; K2 projections and solved-record collection are not included.

Regression: `local_stream2_k1_regression.xml`, 14 passed in 19.94 s, covering
zero/partial/full parent counts plus lookup and neighborhood preparation.
This is CPU TPU-interpreter evidence, not physical TPU compilation or speed.
Collector V4 remains QUEUED at this checkpoint and was not restarted.

### K2 first-hit selection — local only, 2026-09-06

`beam_suffix_hit.pallas_merge_suffix_hit` selects the first successful suffix
when called in ascending BFS suffix order. An immediate/K1 hit (suffix zero)
has priority. Projected solution hashes are separate from unchanged beam
hashes; invalid candidates do not acquire hits. This follows the control order
in read-only `cuda/stream2.cu` (`!found` guard and ascending suffix loop).
It is not yet the full K2 hashing/lookup loop or bounded solved collector.

`local_suffix_hit_regression.xml`: 5 targeted tests passed in 11.37 s, including
the Stream2/K1 composition. CPU interpreter only; no physical compilation,
CUDA execution, whole-suite regression, or performance claim for this addition.
