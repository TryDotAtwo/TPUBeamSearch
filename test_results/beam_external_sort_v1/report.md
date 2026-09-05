# External HBM sort V1: physical correctness accepted

Source: `42037f44ea24e2303ae60464d00cfba33e30fd28`; launcher `8f8804a`.
Kaggle: `trydotatwo/tpu-beam-external-sort-probe`, COMPLETE, retrieved 2026-09-05.

Eight TPU v5 lite devices, one process, JAX/jaxlib 0.10.2 and libtpu 0.0.42.1.
Each device sorts 256 records in uint32 SoA with 11 planes, using 128-column
tiles. The JSON `runs=2` describes two tiles, not two independent repetitions
of the correctness corpus. The fixed RNG seed is 20260905 in the pinned source.

All output elements match the Python lexicographic oracle (zero mismatches).
Actual and expected SHA256:
`bfa5428263d9e4ccc0cd51999e3c1af5b43f4e43f7a26b14c9d5b1bf73f5e655`.
Keys are valid-first, hash words high-to-low, score, payload and original index.
The fixture includes a forced hash duplicate crossing the initial tile boundary.
This gate sorts but does not remove duplicates.

Compilation: 1.065 s. After 3 warmups, 21 synchronized device-resident samples
have median 0.48990 ms, p10 0.46301 ms, p90 0.53411 ms. No matched alternative
or GPU baseline was timed, so this is diagnostic latency, not a speedup.

The source composes 36 compare/exchange passes (8*9/2). HLO preserves Pallas
custom calls for these passes. Each pass stages bounded aligned tiles rather
than the full N=256 candidate array in VMEM. Full-array traffic between passes
is a scaling cost; neither these timings nor HLO establish overlap or efficiency.
The JSON `runs` name is ambiguous and should be clarified in a later harness.

Next gate: thresholding, hash deduplication across tile boundaries, deterministic
compaction/count and neutral tails, followed by larger capacities. Complete
Stream3, full beam correctness, and comparison with Artgor's notebook remain open.
