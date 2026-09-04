# TPU previous-index probe V5

Private Kaggle kernel `trydotatwo/tpu-beam-selector-layout-probe` ran source
`cc32842a6e3b04c0428771382bf63ecf9c09f69d` on all eight TPU v5 lite devices
with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

The probe executes the complete bounded `sort -> previous hash -> uniqueness`
boundary at `N=128`; it is a compiler/correctness diagnostic, not a latency
benchmark.

| previous-index expression | result | mismatch |
|---|---:|---:|
| `maximum(indices, 1) - 1` | compile error at `arith.maxui` | n/a |
| `where(indices == 0, 0, indices - 1)` | exact | 0 |
| `indices - (indices != 0).astype(uint32)` | exact | 0 |

This identifies the concrete V8 `N=128` blocker: Mosaic cannot legalize the
unsigned maximum in the selected VMEM layout. Both equivalent replacements are
physically exact. Production uses the branchless arithmetic form, guarded by a
JAXPR regression that forbids unsigned `max` in bounded dedup. This result does
not address the separate `N=256` multi-source-vreg gather rejection.

