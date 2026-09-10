# Response epoch V3: dynamic slice in chunk preparation

Retrieved complete output to `test_results/beam_response_epoch_v3`.
Source `3fb2250f94c6f6f5a81e174572f4226a8144b466`, process returncode1.
JAX/jaxlib0.10.2, libtpu0.0.42.1, eight distinct TPU v5 lite devices0..7.

Preparation now lowered, compiled and executed for the empty case:
`prepare.mlir` and compiled `prepare.hlo.txt` were saved, and the probe passed
`block_until_ready(prep_exe(...))`. Preparation outputs were not independently
compared at this point, so this is execution evidence, not complete exactness.

First epoch never executed. `step.lower` raises NotImplementedError for
`dynamic_slice` at `beam_final_chunk.py:39`, `starts[peer],counts[peer]`.
Here peer is dynamic and starts/counts are arrays already read from a Ref.
This is a different failure from V1 unsigned reductions and V2 zero-size scan.
Partial JSON has one pending empty epoch0, no actual hashes or mismatches;
there are no successful response epochs, timing or end-to-end beam results.

Next regression should trace the real chunk Pallas body and detect the
unsupported value-array dynamic slice. Replace with supported fixed-lane
selection or direct Ref access, preserving uint32 and all bounds checks.
Do not infer that all dynamic Ref indexing is unsupported from this error.
Require interpreter tests, full regression, then published source pin and one
V4 successor. Do not weaken or reduce the existing eleven-case workload.
