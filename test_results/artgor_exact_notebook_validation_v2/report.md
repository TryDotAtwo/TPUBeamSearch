# Exact Artgor notebook validation v2

Private Kaggle kernel
[`trydotatwo/tpu-artgor-exact-notebook-validation`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-exact-notebook-validation)
version 2 completed on eight TPU v5 lite devices.  The immutable validated
source is `2b99bdf5116f828a21d35b2c5910467f6ab039c2`; the runtime was Python
3.12.13, JAX/jaxlib 0.10.2 and libtpu 0.0.42.1 with x64 enabled.

## Result

All publication gates pass.  The split engine is bitwise identical to
Artgor's original JAX Q forward on both legal scrambles and categorical stress
at 32,768 states per device.  All 54 independently inspected tensor comparison
records have equal reference/candidate hashes, zero mismatches and no mismatch
witness.

| Corpus | Original JAX | Exact split | Throughput | Speedup | Mismatches |
|---|---:|---:|---:|---:|---:|
| Legal scrambles | 24.636 ms | 15.515 ms | 16.896M states/s | **1.588x** | 0 |
| Categorical stress | 24.787 ms | 15.289 ms | 17.146M states/s | **1.621x** | 0 |

This is a paired full-Q inference measurement over a global batch of 262,144,
not a whole-solver speedup claim.

## Search parity and solve gate

- One depth at global beam 1,048,576: all 13 output tensors are hash exact.
- Three consecutive depths: all 39 output tensors, frontiers and packed
  backpointers are hash exact.  The two steady depths give a combined measured
  `1.103x` depth speedup, including inference, routing, communication, dedup and
  top-K at this smaller parity geometry.
- The real gate used global beam 16,777,216 and found pid 1034 at depth 110.
  The final solution contains 116 moves and independently replays to solved.
  Solver wall time was 2,955.75 s (49.26 min); the entire gate took 3,032.65 s.
- There is no paired original-JAX full real solve in this run, so no full-solver
  speedup is asserted.

The downloaded report was checked without trusting its summary booleans.  The
gates were derived again from primitive mismatch counters and hashes; the
competition and puzzle files were downloaded privately, their hashes matched
the run context, and the reported move indices were replayed from the original
pid 1034 state to the solved state.

## v1 failure and fix

Version 1 exposed an x64-only Mosaic compile failure.  The banked embedding LUT
`BlockSpec` returned `(i32, i64, i64)` because two Python zero literals were
promoted when `JAX_ENABLE_X64=True`.  Commit `2b99bdf` makes every LUT index
explicitly int32 and adds a regression test for the generated JAXPR.  Local
verification after the fix: 371 tests passed.  Version 2 then completed the
same frozen protocol without weakening a gate.

## Publication boundary

The public `summary.json` and this report contain no `test.csv` rows, initial
states, solution move string, move-index sequence, or raw private Kaggle log.
Hashes and aggregate validation facts are retained.  Competition data remains
private.
