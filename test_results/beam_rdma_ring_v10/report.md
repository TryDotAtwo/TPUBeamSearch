# Integrated Stream3 split-to-RDMA gate V10

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`1760770fe1898dd225227b5706fb42b73369623d` and failed during Mosaic
compilation, before execution.

Signed masked reductions passed the V9 boundary. Compilation then rejected
`jnp.minimum(start + positions, capacity - 1)` because its `uint32` form lowers
to `arith.minui`, which this Mosaic path could not legalize for the `[8,128]`
vector layout.

All diagnostic indices are bounded by 128, so the clamp now operates exactly
in `int32`; `take_along_axis` accepts the resulting signed indices. No layout,
count or transport semantics changed. Physical V11 confirmation is required.
