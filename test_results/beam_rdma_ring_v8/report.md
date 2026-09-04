# Integrated Stream3 split-to-RDMA gate V8

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`06431387a06e2c042733b51827a02bd5f6cd5d97` and failed during Mosaic
compilation, before execution.

The take-along-axis payload gather lowered past the V7 failure. Compilation
then stopped at `count_ref[0, peer]`: `peer` is device-rank-dependent, and
Mosaic cannot prove that this scalar access into tiled `[1,128]` VMEM is
128-lane aligned (`E2003 CompileTimeMosaicUnprovenMemoryAccessAlignment`). The
same issue applies to `offset_ref[0, peer]`.

The minimal correction loads both aligned control vectors in full, selects the
dynamic peer with a 128-lane equality mask, and reduces the single selected
value. Payload layout, counts and RDMA protocol are unchanged. Physical V9
confirmation is required.
