# Survivor selector layout probe V1

Source `50bdcaf3735d73c926d87786c13db749d5641037`, launcher `a30367a`.
All five isolated cases compiled and executed exactly on eight TPU v5 lite
devices with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

The following forms are physically legal for plain uint32 `[8,128]` data:

- `where` with a broadcast `[1,128]` boolean predicate;
- `where` with an explicitly broadcast `[8,128]` predicate;
- boolean-to-uint32 arithmetic selection in both shapes;
- arithmetic selection with an already-uint32 mask.

Therefore neither `where`, predicate broadcasting, nor boolean conversion is
independently sufficient to reproduce the V7 failure. The invalid layout is
produced by interaction with an earlier operation in the full dedup pipeline,
most plausibly a sort-derived layout. The next probe must bisect the real
pipeline boundaries (first sort, uniqueness, second sort, final survivor
selection); changing the production selector based on this V1 result would be
unsupported.

Compile times are diagnostic only and are not comparable performance samples.
Raw JSON, per-case logs and HLO are retained beside this report.
