# Real dedup stage-bisection probe V2

Source `47b3fba8f85b385e68359884f60f021d3ba13883`, launcher `cd4cd49`.
All five isolated groups returned on eight TPU v5 lite devices with
JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

The initial 11-plane data construction is physically exact. The immediately
following `first_sort` is the first failing boundary: Mosaic rejects a
`select_n` i1-to-i8 vector layout extension. Every later boundary necessarily
contains that first sort and fails with the same signature. This establishes
that V7's failure originates inside the bitonic `_sort` implementation, before
uniqueness, the second sort, final survivor selection, or count output.

The `final_select` case also contains a probe-only `jax.Array.at[].set`, which
is rejected as unsupported Pallas TPU scatter before reaching the already-known
sort failure. That expression is not present in the production kernel and is
not treated as a production defect.

The next probe must bisect a single real compare/exchange: partner gather,
predicate construction, and gathered-data selection (broadcast, row-wise and
arithmetic forms). Production `_sort` remains unchanged until one alternative
is physically exact. No timings in this diagnostic are performance evidence.
