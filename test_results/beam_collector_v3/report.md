# Collector V3: compile rejection, not an execution result

Terminal Kaggle status: ERROR. Source12aae5b085a58ff81eec60ac1eb73009cae927c0,
launcher7c54e20fd1c68135a888ec0dcf1d54f3accdd701. Runtime JAX/jaxlib0.10.2,
libtpu0.0.42.1; eight TPU v5 lite devices0..7. Full output and Kaggle log are
retained here. See `beam_collector_recovery/recovery_bundle.json` and
`full/collector_full.json`, `full/process.log` below that directory.

Only the full collector group ran, returning1 with exact=false. The conditional
integrated S3 group was correctly not started. No executable HLO, execution
correctness result, or latency samples were produced.

Mosaic rejected `tpu.dynamic_gather` with source vector8x256xi32 and indices/
result8x128xi32: `Multiple source vregs along gather dimension`. The earlier
dynamic scalar offset alignment error is absent from this diagnostic. This
does not prove the rest of the collector compiles.

The full collector's partition scatter uses that wide source gather. A new
structural regression fails before the change and detects source gather widths
above128. The candidate correction gathers from individual128-column banks
and selects the matching bank. Seven lowering/scatter tests pass in6.73 s,
including unaligned ranges, cross-bank copies, empty input and fatal overflow.
This is a local candidate, not physical TPU acceptance. It performs N/128 bank
gathers per output tile; no performance or scalability improvement is claimed.

Input SHA256:
`24b304f4b9a47e55c62ba4ab3eb9d494adbb669b5903d0aeec7c419010ed67e7`.
Expected SHA256:
`0e8cceecd17b09ba0444d96e71a8027f65a2a73cf17450bc82c1af09b546e0c3`.
Neither is an actual-output hash. Full local regression82378 completed764 tests
in1066.95 s with no failures/errors/skips before next-source publication/pinning.
See `test_results/local_collector_v4_regression.xml`; physical V4 is still pending.
