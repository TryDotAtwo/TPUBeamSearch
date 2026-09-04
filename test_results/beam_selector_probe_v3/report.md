# First compare/exchange probe V3

Source `0748415e22aebe18ccd85e3f6dad359d772d8ca3`, launcher `95084df`.
All six isolated groups returned on eight TPU v5 lite devices with JAX/jaxlib
0.10.2 and libtpu 0.0.42.1.

The real partner gather is physically exact. The next boundary,
`swap_predicate`, already fails with the V7 `select_n` i1-to-i8 invalid-layout
signature even though it does not select or store candidate data. All four
downstream selection variants inherit that same predicate failure.

This identifies the failing expression as the conditional construction
`swap = where(want_min, ~less & ~equal, less)` inside `_sort`, not partner
gather, predicate broadcasting across metadata planes, or survivor output.
V4 must compare this expression against its boolean-logic identity
`(want_min & ~less & ~equal) | (~want_min & less)` and then exercise one full
compare/exchange. Production remains unchanged until that alternative is
physically exact.

Compile times are diagnostic only and are not performance measurements.
