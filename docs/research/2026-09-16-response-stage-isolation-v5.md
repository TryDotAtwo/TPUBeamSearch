# Response gather isolation V5

Source `c173fbaf974b6ed048f5b6d11a2e3cfd589aca17`, launcher `ee49ac2`.
Artifacts: `test_results/response_stage_isolation_v5/`.
All 17 subprocess reports were downloaded and checked for source,
JAX/jaxlib 0.10.2, libtpu 0.0.42.1, eight distinct device IDs,
logs, lowered MLIR and compiled HLO presence matching compile success.

Eleven stages compile, unchanged from V4. Packing, masked gather,
unmasked gather, bounded gather and composition abort with rc -6.
Rank-two gather returns rc 1 with an explicit Mosaic diagnostic:
`Not implemented: Multiple source vregs along gather dimension`.
The reported operation is `tpu.dynamic_gather` from `vector<32x256xi32>`
with `vector<32x128xi32>` indices along dimension 1, layout `(8,128)`.

Removing the explicit clipping helper does not eliminate the rank-one
layout assertion. Rank-two exposes a separate unsupported lowering instead
of establishing a fix. Do not equate this diagnostic with a proven explanation
of the earlier rank-one assertion.

Next benchmark-only candidate: load each 128-column half separately, gather
from each with indices modulo 128, then select the half using the original
position. Compare with the existing unmasked output, including intervals
whose unused second half is zero, empty peers and prior errors. First prove
interpreter semantics, then physical compilation and execution. The candidate
is not a production change until those gates pass.

No physical response execution, correctness, timing or overlap was measured.
