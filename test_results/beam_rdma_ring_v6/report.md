# Integrated Stream3 split-to-RDMA gate V6

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`3074225c5bb308e295ca8bd074a221e077486819` and failed during Mosaic lowering,
before execution.

The exact failing expression is `remote_ref[:, source_index]` in the wire-slot
adapter. Mosaic TPU rejects advanced integer indexing directly on a Pallas
`Ref` (`ValueError: Cannot do int indexing on TPU`). This is not an RDMA,
synchronization, or correctness mismatch.

The minimal correction materializes the whole aligned `[8,128]` remote block
as a local array using `remote_ref[...]`, then applies the same permutation to
that value. The wire ABI, counts, ring ordering and transport are unchanged.
Physical V7 confirmation is required.
