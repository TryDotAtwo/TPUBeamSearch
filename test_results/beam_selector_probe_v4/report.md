# Boolean swap predicate probe V4

Source `eb9d12ab939b3d8e63524742788bb4cbe65af7e3`, launcher `72a23aa`.
Eight isolated cases ran on eight TPU v5 lite devices with JAX/jaxlib 0.10.2
and libtpu 0.0.42.1.

The original conditional swap predicate again fails at `select_n` with the
i1-to-i8 invalid layout. Its mathematically equivalent pure boolean form

```
(want_min & ~less & ~equal) | (~want_min & less)
```

is physically exact. A complete partner-gather plus compare plus selection
using that predicate is also physically exact. This closes the attribution:
the unsupported operation is the conditional boolean `where` used to construct
the bitonic direction-dependent swap predicate, not gather or data selection.

Production `_sort` now uses the proven boolean identity. A regression rejects
boolean `[128]` `select_n` swap predicates. Targeted dedup, Stream3 and original
C++ source-oracle tests pass, followed by the full local suite: 581 passed and
5 skipped in 264.60 s. Physical full-kernel confirmation remains required.
Probe compile times are diagnostic only.
