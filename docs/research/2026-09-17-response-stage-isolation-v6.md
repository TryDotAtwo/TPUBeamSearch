# Response gather isolation V6

Source `830389d9567d4398ad583a5a85b58a36dee9b4d2`, launcher `db3e6e1`.
Artifacts: `test_results/response_stage_isolation_v6/`.

All 18 nested reports were downloaded and checked for source, JAX/jaxlib
0.10.2, libtpu 0.0.42.1, eight TPU v5 lite IDs, process logs, lowered MLIR,
and compiled HLO matching the reported compile result.

The new split-gather candidate compiled successfully. It loads the two
128-column halves separately, performs two rank-2 gathers with modulo-128
indices, and selects the half from the original position. This avoids the
unsupported `dynamic_gather` from a 32x256 vector diagnosed by V5.

The original masked/unmasked and bounded rank-1 variants still abort; the
rank-2 full-width variant still returns Mosaic's multiple-source-vregs error.
Packing and composition remain aborting. Split-gather is therefore a valid
next physical-execution candidate, not yet a production fix.

No response execution, numerical correctness, latency, overlap or end-to-end
beam speed was measured by this compile-only run.
