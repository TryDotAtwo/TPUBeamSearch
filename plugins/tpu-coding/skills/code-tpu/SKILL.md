---
name: code-tpu
description: Use when writing, debugging or benchmarking Python/JAX/Pallas programs on TPU, investigating TPU numerical or memory regressions, or maintaining evidence from TPU experiments. Not for CUDA-only tuning.
---

# Code TPU

Treat a TPU implementation as three separate contracts: the model/task, its
floating-point evaluation, and the measured execution scope. This reference
combines primary documentation with scoped experiments; it does not select
Pallas over JAX in advance.

## Route by the current question

Read the relevant reference before changing the corresponding implementation.
Do not load the entire library for a narrow question.

| Question | Read |
|---|---|
| Which TPU, memory capacity, API or interpreter? | [Runtime and hardware](references/runtime-hardware.md) |
| Shapes, padding, BlockSpecs, MXU tiling, VMEM, DMA or fusion? | [Layout and pipelines](references/layout-pipelines.md) |
| Dense/LN drift, JIT boundaries, model or beam correctness? | [Numerical validation](references/numerical-validation.md) |
| Which implementation wins, timing, multi-device scaling or Kaggle? | [Benchmarking and scaling](references/benchmarking-scaling.md) |
| What did TPUBeamSearch actually establish? | [Scoped case studies](references/case-studies.md) |
| New measurement, contradiction or runtime migration? | [Evidence maintenance](references/maintenance.md) and [evidence records](references/evidence.json) |

## Establish the working contract

Recover the exact architecture and consumer from code: logical, storage and tile
sizes; encoding; normalization width/epsilon; residual and activation order;
parameter layout; precision/rounding; output count; masks and score direction.
For Q inference, `MOVE_COUNT` scores per parent are not separate child forwards.
Preserve a working BN path when experimenting with LN.

Use JAX for orchestration, compilation and a reference; choose Pallas kernel
boundaries from measured bottlenecks and live memory. One kernel per stage,
layer or block is an experiment, not a universal target. When valid performance
is indistinguishable within measurement uncertainty, prefer the simpler code.

## Frequent misreadings

- A requested accelerator or scoped VMEM limit is not complete hardware inventory.
- A passing interpreter is not a TPU compilation or performance result.
- Zero-padding a Dense reduction and padding a LayerNorm population are different.
- Argmax agreement is not the quality gate for a minimizing global Q-beam.
- Matched aggregate errors do not establish pairwise tensor equality.
- A faster block, more buffers or expert agreement does not prove a faster full model.

Use available project experts for unresolved domain questions with sufficient
context; verify advice against code, primary sources and experiments. This
plugin grants no new authority for remote runs, publication or data disclosure.
