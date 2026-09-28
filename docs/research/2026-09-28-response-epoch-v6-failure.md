# Response epoch V6: pre-exchange byte corruption

Private Kaggle `trydotatwo/tpu-beam-response-epoch-gate` V6 ran pinned source
`415ae3524aad20f8f82c7b4247be608af2681234` on eight distinct TPU v5
lite devices (JAX/JAXLIB 0.10.2, libtpu 0.0.42.1). It ended `ERROR`;
`process.json` reports return code 1 and `response_epoch.json` reports
`exact: false`. All 18 output/log/HLO/MLIR files (3,118,650 bytes) are in
`test_results/beam_response_epoch_v6/`.

Isolated packing and composition compiled, and the three `empty` epochs were
exact. The first `self` epoch 0 was not exact: per-device wire byte mismatch
counts were `[2064,2097,2052,2000,1968,1967,1968,2011]`, with zero
control mismatches. Its grouped live-word mismatch counts were
`[2096,2097,2084,2032,2000,1999,2000,2043]`; its packet-word counts
matched the output mismatch counts. Packet controls and preparation errors
were zero. The first output mismatch was device 0, row 1, byte column 2,
actual 0 versus expected 102. The byte-column histogram was nonzero only
at columns congruent to 2 modulo 4. Later nonempty epochs/fixtures did not
run; the 11-fixture/33-epoch gate has not passed.

This places an observed divergence in the grouped words before exchange,
but does not yet distinguish the wire-to-planes conversion from grouping.
The packet and output diagnostics also cannot prove that no additional error
occurs later. An isolated physical probe of each boundary is required before
changing production.

The V6 `interval_mismatches` values `[7,6,5,4,3,2,1,0]` were a diagnostic
oracle bug, **not** evidence of wrong production intervals: for a self-routed
rank with 129 live rows, every following peer has start 129. A regression
test with this expected prefix failed before the oracle repair and passes
after it. This repair changes diagnostics only; no production fix is claimed.

No response correctness, full beam correctness, or throughput claim follows
from V6.
