# Final V2: parent DMA alignment rejection

Source `cec9e41940a95c2388fc84446a9bdd6d32649244`, launcher `e1a6fc3`.
Original outputs, nested JSON, HLO and logs: `test_results/beam_final_v2`.
The coordinator returned ERROR: exchange16/16 and coverage7/7 exact, both
return0; CUDA fixture materialization returned1 at compilation of count0_remote.
Five remaining materialization cases were not reached. No latency measurement.

The V1 output BlockSpec rejection is no longer the reported failure. Mosaic
now rejects the parent DMA at `copy.start()` because a dynamic row offset into
`memref<7x128xi8>` with tiled layout `(8,128)(4,1)` is not aligned to eight rows.
Arbitrary parent IDs cannot truthfully satisfy `assume_multiple(...,8)`.

The V3 candidate moves the record dimension outside the two minor dimensions:
internal parents and output are `[records,1,width]`, VMEM is `[1,1,width]`.
External two-dimensional arrays and exact response bytes are unchanged. Parent
and output DMA still wait before stage reuse. No alignment assertion is added.
This is a layout candidate; physical padding/conversion and TPU acceptance
are unverified. The reshape is not claimed to be zero-copy.

A structural regression failed on the original `[7,128]` call input before
implementation. Three focused tests passed in4.85s. Full local regression with
both C++ oracles:937passed750.09s, no errors/failures/skips, recorded in
`test_results/local_final_record_axis_full.xml`. CPU tests/JAXPR do not establish
TPU compilation, CUDA execution, full beam correctness, or speedup.
