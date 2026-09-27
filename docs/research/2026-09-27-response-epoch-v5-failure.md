# Response epoch V5: first physical nonempty mismatch

Pinned source `d5c8a45f02a9a71e05da2f5584c887fbf968bda3` ran on eight TPU v5 lite devices with JAX/JAXLIB 0.10.2 and libtpu 0.0.42.1. The Kaggle kernel ended `ERROR`; its bundle's `all_exact` is false.

The isolated packing and composition executables both compiled. These are compile-only gates, not evidence of physical correctness. The full response probe passed all three empty epochs with exact uint8 wire and uint32 control shapes, values, and hashes. The first nonempty `self` epoch 0 failed: wire mismatch counts by rank were `[2064,2097,2052,2000,1968,1967,1968,2011]`; all eight control mismatch counts were zero. Epochs 1 and 2 and the other nine fixtures were not executed. There are no timing results.

The physical result establishes a response-byte corruption on this composition, not its source. In particular, the log alone cannot distinguish grouping, peer packing, local self-copy, receive compaction, or planes-to-wire conversion. A local interpreter regression with 129 shuffled records crossing the 128-record boundary passes all three epochs; it does not close the physical gate.

The next pinned run is diagnostic, without production changes. On the same first failing case it records preparation live-word and rank-interval mismatches, packet word/control mismatches before exchange, output mismatch histogram by byte column/row, and first differing byte coordinates. The next fix must be based on the first divergent physical boundary and a failing regression test; it must not be inferred from the approximate one-eighth output mismatch rate.

Artifacts: `test_results/beam_response_epoch_v5/beam_response_epoch/bundle/followup.json`, nested `full_response/response_epoch.json`, full Kaggle log, process logs, and compile HLO/MLIR in the same directory.
