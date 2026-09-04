# Eight-TPU two-slot RDMA gate V2

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`676931d21d60d34fbd693ee89cc89520a03c517e` on all eight TPU v5 lite devices
with JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.

Both isolated physical cases completed and matched independent host oracles:

| case | active epochs | slots | mismatches | median diagnostic call |
|---|---|---:|---:|---:|
| all active | 0,1,2,3 | 2 | 0 | 0.519 ms |
| alternating zero-count | 0,2 | 2 | 0 | 0.501 ms |

The kernel executes destination readiness before each active send, distinct DMA
send/receive semaphores per slot, explicit send and receive waits, a local copy
that models receiver consumption, and a regular-semaphore acknowledgement
before slot reuse. Four epochs force both slots to wrap once. Inactive epochs
advance the program and write a known-zero archive without starting or waiting
on remote DMA; no phantom semaphore state or hang occurred.

This establishes the bounded synchronization and slot-lifetime mechanism. It
does not yet establish per-edge variable counts, Stream3 routing, capacity
handling, sustained bandwidth, or communication/compute overlap.

