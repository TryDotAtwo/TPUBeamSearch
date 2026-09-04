# Stream3 variable-count RDMA gate V4

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`d415d9792c47873a5b4e5521d049bfa2e730e963` on eight TPU devices and failed
after successful compilation, during the benchmark warmup invocation.

The compiled executable has four positional inputs, but the shared timing loop
called it with the placement tuple as one positional input. JAX rejected the
input pytree before any timed execution. The initial correctness invocation in
`run_variable_exchange` already used `executable(*placed)` correctly; only the
generic warmup/timing path was wrong.

The minimal correction centralizes compiled invocation in `call_compiled`,
which splats tuple placements and preserves the single-input call path. A unit
test covers both shapes. V5 must still physically validate Pallas lowering,
the full variable-count protocol, hashes, neutral tails, and timing.
