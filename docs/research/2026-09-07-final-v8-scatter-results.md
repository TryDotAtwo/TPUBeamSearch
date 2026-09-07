# V8 scatter compile rejection

Private final-gate V8 terminated ERROR. Source
580a05c6f651eb685698fb49a9b5a9065d628292, launcher 1f36ce2.
Artifacts: `test_results/beam_final_v8_scatter/`.

Both isolated children returned 1 (not a native signal), leaving only
`count0_valid` in compiling status. No case executed, no correctness or timing
result exists. Both report JAX/jaxlib 0.10.2, libtpu 0.0.42.1 and eight distinct
TPU v5 lite devices. Coordinator correctly reports all_exact=false.

The compiler identifies `beam_final_scatter.py:36`, `load.start()`, and prints
the failing operation: slicing `memref<256x128xi8>` with tiled layout
`(32,128)(4,1)` into `memref<1x128xi8>` at a dynamic first-axis offset.
It cannot establish offset divisibility by 32. This is concrete load-side
layout evidence, not attribution solely from a stack trace. Count zero still
compiles the conditional branch. Arbitrary record indices are not multiples
of 32; assume_multiple would be false and is not an acceptable fix.

Next candidate: explicit record-axis internal HBM/staging layout, preserving
external wire/frontier ABI and arbitrary target indices, as for materialize.
Test both read and write mapping; load rejection currently hides any later
store rejection. Add a regression before changing the kernel and confirm on
physical TPU. Do not claim interpreter acceptance proves TPU compilation.

Local response-preparation full run 53469 is no longer addressable, and its
expected XML is absent. Its earlier partial progress is not a passing full
regression. Revalidate OS process ownership before deciding whether to rerun.
