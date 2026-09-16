# Response stage isolation V4

Source: `57e45ca790ac185ebe77fa0a5a4c766e44070540`; launcher: `affb74e`.
Artifacts: `test_results/response_stage_isolation_v4/`.

Kaggle finished ERROR. All 15 subprocess reports and logs were downloaded;
each report identifies JAX/jaxlib 0.10.2, libtpu 0.0.42.1 and eight distinct
TPU v5 lite devices. Lowered MLIR exists for every case, compiled HLO for
exactly the successful cases. The coordinator returned 1.

Eleven cases compiled: control, selection, guard, first DMA, second DMA,
row copy, positions, clipped positions, exchange, receive and planes-to-wire.
Four aborted with return code -6: packing, masked gather, unmasked gather,
and composition. The unmasked gather log retains the layout-rank assertion
`1 vs. 2`.

## Attribution boundary

Row extraction and aligned row storage compile in the row-copy control.
Dynamic position outputs compile with and without explicit clipping.
Adding the clipped gather to the DMA-backed row fails even without the
length mask. Thus the mask is not necessary for this failure. This does not
yet identify a single source expression: the gather's internal lowering and
its surrounding layout interaction remain to be separated.

Position probes do not consume DMA data; dead DMA can disappear. Their legal
indices already fit 0..255, so a successful clipped-position probe does not
prove that clipping survives optimization. Retained-IR inspection remains
necessary before attributing an optimized-away operation.

Next diagnostic should compare rank-one and rank-two gathers with identical
runtime indices and payload, plus an explicitly bounded gather without the
helper's clipping. Keep production unchanged until the narrower diagnostic
supports a minimal fix; then run full regression with both C++ oracles.

These are compile-only observations: no executed response epochs, numerical
correctness, timings, overlap or full-beam speedup are established by V4.
