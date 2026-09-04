# Eight-TPU Pallas RDMA ring probe V1

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`67a9a40d7030bb865a57acf77d8cfe7e229e738b` on all eight TPU v5 lite devices
with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

The first physical transport gate is exact: every device pushed its local
`uint32[8,128]` shard to its right neighbor through Pallas remote DMA using
distinct send and receive DMA semaphores and the explicit sequence
`start -> wait_send -> wait_recv`. The output has zero mismatched elements and
its SHA-256 equals the independent host rotation oracle.

The synchronized already-compiled call median was 0.521 ms (p10 0.484 ms,
p90 0.587 ms; 3 warmups and 21 samples). This is a one-hop 32 KiB/device
diagnostic, not complete Stream3/S5 communication throughput and not overlap
evidence.

Still required: destination readiness, two-slot reuse/ack, zero-count epochs,
variable-count Stream3 exchange, and a physical race/overlap profile.

