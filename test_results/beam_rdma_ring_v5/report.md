# Stream3 variable-count RDMA gate V5

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` completed on eight
TPU v5 lite devices with source
`2f4cb6fa70194282e68e763ae957bb6bd3c0dd07`.

The isolated seven-epoch variable-count exchange is exact:

- payload and count mismatches: `0`;
- payload SHA-256 equals the independent oracle;
- count SHA-256 equals the independent oracle;
- asymmetric counts include zero-length sends and one capacity-bound send of
  128 records;
- the full fixed-capacity payload comparison also validates neutral tails;
- two physical receive slots are reused across seven peer offsets.

Compilation took 0.469 s. After three warmups, 21 synchronized diagnostic
samples had median 0.6545 ms, p10 0.61693 ms and p90 0.7064 ms. This is an
isolated transport measurement, not a beam-search or inference speedup.

This closes the bounded variable-count transport primitive only. Stream3 still
requires one compiled split-to-wire-layout-to-exchange gate before it can be
marked complete.
