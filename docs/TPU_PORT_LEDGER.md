# Whole-architecture TPU port acceptance ledger

The user's completion condition is the whole architecture plus a GPU/TPU
comparison, not isolated kernels. No complete-port or performance claim yet.

| Contract | Implementation/evidence | Remaining acceptance |
|---|---|---|
| Logical types / uint32 SoA | beam_types.py; high-word and padding tests | all consumers, physical TPU |
| Local pipelined transport | beam_transport.py, 2/3 input buffers; race simulation | physical DMA trace and throughput |
| S1 MOVE_COUNT score producer | existing inference engines only | exact selected-model integration, quantization without HBM float-Q |
| S2 immediate hash / exact goal | beam_stream2.py; independent C++ source oracle | real TPU compile, valid-input launcher, K1/K2, bounded solved collection |
| Hash128 owner/shard arithmetic | beam_hash.py uint32 pairs; edge/random modulo tests and C++ oracle | physical TPU lowering, fingerprint and K1 bucket salts |
| S3 threshold/sort/dedup | beam_dedup.py diagnostic bitonic baseline; payload tie-break tests | routing/compaction buffers, source Stream3 full split parity, HBM-scale sort |
| S4 threshold/sort/dedup | same primitive with score/parent64/route tie-break; C++ oracle | resident A/B collector, independent capacity/trigger, committed histogram |
| S5 | ownership/epoch design only | remote DMA readiness/ack, zero counts, coordinated threshold, race and TPU tests |
| Three scratch overlays | design only | explicit arena plan, alias report, drain gates |
| Final | source read, design only | exact global cap/ties, balance, request/response, padding, history, replay |
| Stop | design only | bounded solved records, no collective deadlock, multi-rank stop |
| Whole depth / multi-depth | not implemented | original CUDA replay and 8-TPU replay on identical fixtures |

## Evidence boundaries

2026-09-03 local verification: 535 tests passed in 158.84 s with
BEAM_SOURCE_ORACLE enabled, including both original C++ differential tests.
No skips in this run. JAX 0.10.1 CPU interpretation; no physical TPU or CUDA
execution evidence for these new primitives yet.

The adapter `tests/beam_source_oracle.cpp` links original `src/hash.cpp`,
`src/state.cpp`, `src/stream4.cpp` from D:/100XH100 read-only. It executes on the
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
- cuda/stream2.cu: d52252daba39fc913a31c7ded25f08721b8b58aa8c5ca0e09872bf7a6e61f126

## Next execution order

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
samples; only exact cases are eligible. No V2 TPU result is established yet.

V2 is now COMPLETE; report: `test_results/beam_primitives_v2/report.md`.
Five exact cases; six compile failures moved past gather to scalar uint8 extract
(Stream2) and sentinel scatter (dedup). Packing medians serial/pipeline:
b2 1.347320/0.670960 ms (2.008x), b3 1.335930/0.632890 ms (2.111x), at 65536
candidates/device. These are primitive-call measurements, not beam/inference
speed or profiler-confirmed overlap. V3 candidate fixes are in regression testing.

1. Validate these primitives on physical TPU before propagating their lowering
   choices. Preserve the already active V10 session; never run a second TPU job.
2. Add a standalone remote-DMA ring test: readiness, send/receive wait, slot
   acknowledgement, zero-count and multiple-wrap cases, serialized control.
3. Expand source oracle to Stream3 grouping and final request/response. Build
   HBM-scale sort/merge and preallocated A/B collector with fatal overflow.
4. Integrate K1/K2, histogram epochs, final phases and stop, one tested subsystem
   at a time. Keep every row above explicit until its acceptance evidence exists.
5. Compare discrete search with identical score tensors first. Then compare actual
   inference separately; FP16/BF16 differences cannot excuse a search logic defect.
6. Compare whole GPU/TPU depths, selected identities, hashes, parents/routes,
   history replay, overflows, ties, valid counts and solutions; separately measure
   runtime and overlap. Record any allowed layout/scheduling changes explicitly.
