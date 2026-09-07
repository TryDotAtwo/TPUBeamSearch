# V4 native abort: isolation protocol

Observed source: b4beacd95d5de126ef3ad6ac503e44b72585ca60.
Artifacts already downloaded: test_results/beam_final_v4. Materialization
subprocess returns -6 at first compilation; exchange16 and coverage7 remain
exact. Assertion `layout.h:341 arr.size() >= layout_rank(implicit_dim)` reports
1 versus2. This does not identify a Python expression. No production change
is justified solely by that stack.

## Small sequential subprocess bundle

Keep the coordinator free of JAX imports. Each case must run in a fresh process,
save runtime/source/input hashes and a pending JSON before compilation, save
lowered IR before backend compilation where available, then compiled HLO and
exact output after execution. Parent coordinator retains returncode and partial
JSON after SIGABRT and continues the remaining cases. A crash is not a numerical
failure and must not erase subsequent evidence.

Use the same seven-parent uint8 fixture, runtime indices selecting parent6 and
parent0, width128, reversed valid permutation, output capacity128. Match V4
JAX_ENABLE_X64=False and runtime versions. Do not use assume_multiple. First
run on eight physical TPU devices with identical local inputs, not CPU-only.

Cases, each differing by one operation from the preceding scope:

1. Record-axis `[7,1,128]` parent DMA into `[1,1,128]` VMEM and output DMA,
   no arithmetic. This tests the singleton layout without permutation.
2. Same DMA path plus uint8->int32->uint8 conversion, no gather.
3. Same plus the V4 rank-one int32 gather and narrowing, no response packing.
4. Same permutation represented as rank-two data/indices `[1,128]` throughout
   gather; compare to3 as a layout experiment, not an assumed fix.
5. Same as3 plus zero-tail and four-byte little-endian response index packing.
6. Full unchanged production materialization on the same fixture, including
   request validation, scalar extraction, dynamic move selection and branch.

Test both a valid record and zero count. Zero count still compiles the branch;
it cannot demonstrate a valid parent read or child byte generation.
Use explicit NumPy expected bytes, never a second copy of the kernel as oracle.
Expected padding, target bytes and invalid-count/first-invalid sentinel are
checked separately. Include all byte values in local interpretation coverage.

## Attribution rules

If1 crashes, later operations are not implicated; isolate output versus input
DMA and singleton shape next. If2 first crashes, isolate conversion/layout.
If3 crashes and4 compiles, there is evidence for rank-sensitive gather lowering,
but recheck full production before attributing the whole failure to gather.
If5 or6 first crashes, isolate packing or production-only control/data flow.
Passing an extracted probe does not prove production or full beam correctness.

No timings or speed claims in this bundle. Preserve physical allocation/layout
conversion evidence; `[records,1,width]` is not presumed a free reshape. After
local tests and publication, launch one private TPU session. Do not re-run V4
unchanged or rerun accepted S5 tests to substitute for isolation evidence.
