# Integrated Stream3 split-to-RDMA gate V11

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` completed source
`2a36db2797f02d1d0f646a65e96c09f9989f32ec` on eight TPU v5 lite devices.

The single compiled `pallas_stream3_split → ring wire slots → variable RDMA`
program is exact against the independent host oracle:

- mismatched elements: `0`;
- combined local, wire and received SHA-256 equals the expected SHA-256;
- all eight device-derived ranks are covered;
- seven peer-offset epochs are covered;
- rank 0 sends the capacity boundary of 128 candidates to one remote peer;
- other ranks exercise mixed local/remote records and zero-count edges;
- local counts, wire counts, received counts, route origin/owner/move fields and
  full neutral tails are included in the combined comparison.

Compilation took 0.948 s. After three warmups, 21 synchronized samples had
median 0.80834 ms, p10 0.77019 ms and p90 0.85918 ms. This timing covers the
bounded N=128 split, wire adapter and transport together; it is not an
HBM-scale Stream3, inference, or full beam-search performance claim.

The bounded split-plus-transport correctness gate is now closed. HBM-scale
partition/sort/merge remains open.
