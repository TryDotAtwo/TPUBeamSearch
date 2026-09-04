# TPU beam primitive gate V10

Private Kaggle kernel `trydotatwo/tpu-beam-primitive-compile-and-correctness-gate`
ran source `ecca3b38ec846d716ddc2c6fa21b7ab691888e28` on all eight TPU v5 lite
devices with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

The aligned host oracle confirms value-level physical correctness for both
bounded `N=128` dedup variants:

| case | result | mismatches | median diagnostic call |
|---|---:|---:|---:|
| Stream3 dedup | exact | 0 | 0.661 ms |
| Stream4 dedup | exact | 0 | 0.672 ms |

The logical count is stored in lane zero of the aligned `[1,128]` control
plane; all unused lanes match zero exactly. This closes the previous-index and
oracle-shape regressions for the bounded primitive. The timings are independent
cross-process diagnostics, not a matched Stream3-vs-Stream4 A/B and not a
whole-beam performance result.

Four packing variants, routing and both split cases also remain exact. Matched
packing medians at 65,536 candidates/device are 1.344/1.345 ms for serial b2/b3
and 0.669/0.623 ms for pipeline b2/b3. Cross-run variation must not be treated
as an additional speedup claim.

Remaining independent compiler blockers are unchanged:

- `N=256` dedup: multiple source vregs along the gather dimension;
- hash120: native VectorLayout join abort;
- hash150: unsupported uint8 reshape `[8,160] -> [1280]`.

