# Integrated Stream3 split-to-RDMA gate V9

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`917fb15265bd717625962718b27353f47b83def4` and failed during Mosaic lowering,
before execution.

Aligned full-vector control loads and peer masking passed the prior alignment
boundary. Mosaic then rejected the masked `uint32` sum because unsigned integer
reductions are not implemented in this backend. The existing Stream3 split
already uses the supported form: reduce as `int32`, then cast the bounded result
to `uint32`.

The wire adapter now applies that same arithmetic to count and offset
selection. Values are bounded by the diagnostic capacity 128, so the signed
intermediate is exact. Physical V10 confirmation is required.
