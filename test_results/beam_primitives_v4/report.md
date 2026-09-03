# V4 isolated TPU primitive gate

Source ea3eaa09822777e47c65101299fac7093bb8d952, launcher ee49e25.
Kaggle COMPLETE, but all_exact=false. All eight groups record eight TPU v5 lite
devices, JAX/jaxlib 0.10.2 and libtpu 0.0.42.1. Full output and process logs
are preserved beside this report.

## Correctness and compiler results

Four pack cases and owner routing are exact. Isolation works: hash_120 exits
with SIGABRT (-6), but all subsequent cases are still attempted.

- hash_120: repeated native VectorLayout::join / inferElementwise abort. The
  stack does not identify its responsible source expression.
- hash_150: specific unsupported shape cast at beam_stream2.py parent flatten:
  `vector<8x160xi8> -> vector<1280xi8>`. No output was executed.
- Four dedup cases: `Reductions over unsigned integers not implemented` at the
  final survivor count `sum(keep.astype(uint32))`. This is distinct from the
  already-removed gather/scatter problems; no successful physical sort result
  can be claimed until lowering and execution complete.

## Matched packing timing

65536 candidates/device, 524288 across eight devices. Three warmups, 21
forward/reverse alternating synchronized calls; placement and compile excluded.

| Configuration | Median ms | p10..p90 ms | Aggregate M candidates/s |
|---|---:|---:|---:|
| serial b2 | 1.360760 | 1.331460..1.376010 | 385.291 |
| serial b3 | 1.333540 | 1.310620..1.379530 | 393.155 |
| pipeline b2 | 0.672240 | 0.636760..0.690760 | 779.912 |
| pipeline b3 | 0.633080 | 0.614460..0.658630 | 828.154 |

Median ratios are 2.024x (b2) and 2.106x (b3). The distributions overlap for
pipeline b2/b3, so do not infer a universal buffer-count winner. Routing at
256 hashes/device has diagnostic median 0.510899 ms, p10..p90
0.485130..0.551770 ms. Cross-process timings are not matched comparisons.
These remain packing-call results, NOT inference/beam speedup or DMA-overlap
profiling evidence.

## Next changes

Dedup count is bounded by 4096: signed int32 reduction followed by uint32 cast
preserves every value and avoids the explicitly unsupported unsigned reduction.
A structural regression fails before this change and passes after it; physical
confirmation remains required. Apply the same bound to the new bounded Stream3
split counters. Preserve the unresolved hash cases as diagnostic failures, not
silently corrected or excluded from the full gate.
