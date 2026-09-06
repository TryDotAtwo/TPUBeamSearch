# Original CUDA final response binary oracle

`validation/cuda_final_oracle.cu` is a standalone adapter linked to the unchanged
`D:/100XH100/cuda/final_materialize.cu`. Source SHA remains
`c3578668299fd2883ef9a0dda336e7bba1376f4ed44fad613a0867ad041585ea`.
Build: CUDA12.5/MSVC, sm86, C++17, state120/storage128/alignment16/MOVE_COUNT24,
BEAM_DEBUG_FINAL_VALIDATE=1. Outputs in `.local/cuda_final`; no source writes.

## Binary fixture

Little-endian magic `TFIN0001`, six uint32 values (parent count, request count,
move count, logical bytes, storage bytes, target count), then parent rows,
16-byte FinalRequest records and generator rows. Responses are raw storage128
rows. The adapter bounds allocation sizes and validates generator permutations
and request memory bounds before invoking original CUDA functions.

## Important validator distinction

The first run (`cuda_final_oracle_v1`) rejected reversed target indices with
reason_slot=1. This was an adapter contract mistake, not a byte mismatch.
Original dispatcher calls the debug `target==slot` validator on its local
materialization path (around4848). Its remote response path (around4605) calls
`final_materialize_responses_cuda` without that local-only requirement.
The corrected adapter mirrors these modes: `--local-slots` enables the original
GPU slot validator, while remote mode allows reordered targets and retains host
memory-bounds checks. The original CUDA functions were not modified.

## Actual GPU results

`validation/cuda_final_oracle_smoke.py` creates deterministic seed609 fixtures
with all24 cyclic generators and an independent NumPy byte oracle.
On RTX3070 Laptop, driver572.70, all six cases in `cuda_final_oracle_v2` passed:
remote counts0,1,127,128,129 and local count127. Every output byte and SHA256
matches, including target index and remaining padding; every process exit0.
Input/output binaries and JSON retain the exact fixtures for TPU reuse.

This is actual single-GPU materialization evidence, not TPU execution,
distributed routing, full beam selection, multi-depth replay or a speed claim.
The v1 failure is retained rather than overwritten by the v2 successful run.

`validation/compare_cuda_final_pallas.py` subsequently loads the identical
saved binaries (verifying input and CUDA output hashes), converts request AoS
to aligned Pallas SoA and calls the interpreter. All six response hashes match
the actual CUDA outputs and padded extra rows are zero. Evidence is
`pallas_interpreter_comparison.json`. This adds common-fixture arithmetic/layout
evidence, but still does not constitute physical TPU execution.
