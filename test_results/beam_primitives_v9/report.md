# TPU beam primitive gate V9

Private kernel `trydotatwo/tpu-beam-primitive-compile-and-correctness-gate`
ran source `df42d79cbf1a70a8a96dcb4d09fede80b844eed4` on eight TPU v5 lite
devices with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

V9 physically confirms that the branchless previous-index replacement removes
the `N=128` `arith.maxui` compiler failure: both Stream3 and Stream4 bounded
dedup now compile and execute. They were reported as `structure_mismatch`, not
value mismatch, because the benchmark oracle still described the old logical
count shape `[1]`; the production kernel has intentionally returned an aligned
`[1,128]` control plane since V5, with the logical count in lane zero and zero
elsewhere. A failing regression reproduced this oracle defect, and the oracle
is now aligned with the production ABI. Physical exactness therefore remains
to be confirmed by V10.

Other results:

- four pack variants, routing, and both split cases: exact;
- `N=256` Stream3/Stream4 dedup: unchanged compile rejection, multiple source
  vregs along the gather dimension;
- hash120: unchanged native VectorLayout join abort;
- hash150: unchanged unsupported uint8 reshape `[8,160] -> [1280]`.

Matched pack medians at 65,536 candidates per device were 1.206 ms / 1.049 ms
for serial b2/b3 and 0.594 ms / 0.578 ms for pipeline b2/b3. These numbers are
primitive-call measurements only and do not establish beam-level overlap.

