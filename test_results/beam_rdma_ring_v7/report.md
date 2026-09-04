# Integrated Stream3 split-to-RDMA gate V7

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`fc8b524b61dc66528e94e177d3d216dfeed09dbc` and failed during Mosaic lowering,
before execution.

Materializing `remote_ref[...]` fixed direct Ref indexing, but
`remote_value[:, source_index]` lowered to a general gather. Mosaic TPU only
supports take-along-axis-shaped gathers here and rejected the expression with
`NotImplementedError: Only take_along_axis-like gathers supported`.

The minimal correction broadcasts the one-dimensional source index to the
full `[8,128]` value shape and uses `jnp.take_along_axis(..., axis=1)`. The
selected elements and every downstream wire/RDMA contract are unchanged.
Physical V8 confirmation is required.
