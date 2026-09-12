# Response epoch V4: native vector-layout compilation abort

Source: `2a7f7f594cf44c5756c18465df649028ee06e822`; launcher: `b898282`.
Complete available output was retrieved to
`test_results/beam_response_epoch_v4_diagnostic/` after network retries.
The earlier `beam_response_epoch_v4/` directory contains only a partial download.

## Confirmed evidence

- Process return code: `-6` (SIGABRT).
- JAX/jaxlib `0.10.2`, libtpu `0.0.42.1`; eight distinct TPU v5 lite devices, IDs 0–7.
- Preparation lowered, compiled and executed for the empty fixture. Its outputs
  were not independently compared, so this is not an exactness result.
- `epoch.mlir` was saved after `step.lower`. The process aborted during
  `lowered.compile()`, before the first epoch executed; no compiled epoch HLO exists.
- Fatal assertion: `arr.size() >= layout_rank(implicit_dim) (1 vs. 2)` in
  `VectorLayout::join`, with `VectorLayoutInferer::inferElementwise` on the stack.
- Partial JSON contains only pending `empty`, epoch 0: an expected hash but no
  actual hash or mismatch counts. Its `exact=false` is not a numerical mismatch.
- No completed response epochs, timing samples, or end-to-end beam evidence.

## Attribution boundary and next experiment

Unlike V3, value-array dynamic slicing no longer prevents epoch lowering.
This does not establish which expression causes the new compiler abort.
The native stack alone cannot identify the source operator.

Isolate compilation in separate subprocesses so one native abort cannot hide
later diagnostics: peer packing, exchange, receive compaction, planes-to-wire,
then the unchanged composed epoch. Preserve production shapes and dynamic
arguments. Receive compaction itself contains control, prepare, external sort
and finish kernels; isolate those further if that stage fails. Save precompile
MLIR, return codes and logs for every case. Only after a failing expression is
reproduced should a minimal production fix and regression test be introduced.
CPU interpretation or JAXPR inspection is not physical TPU acceptance.

The production eleven-fixture, three-epoch gate remains required after repair;
do not replace it with compile-only success or weaken its correctness checks.
