# Exact eight-TPU inference frontier

## Objective

Improve the accepted exact Artgor LayerNorm ResMLP inference path without
changing the checkpoint, Q-only graph, BF16 model contract, eight-way batch
sharding, or the real device-resident split after the final residual block.
Beam expansion, top-K, deduplication, and inter-device beam communication are
out of scope.

The accepted control is commit
`7865ac455b0d4dfbb3d6e8b68430164790fc076c`:

1. prepacked FP32-stored, BF16-valued embedding banks;
2. Pallas flat embedding plus the unchanged JAX input and ten-block ResMLP;
3. a real BF16 hidden boundary after residual block 9;
4. a separately compiled JAX `1024 -> 30` head.

## One-job staged matrix

The private Kaggle job records failures incrementally and keeps one TPU
session active.

1. Compile and profile original `jax_model.apply`, typed JAX, the accepted
   prefix, the accepted JAX head, and the composed two-dispatch control.
2. Sweep prefix `BM = 2048, 4096, 8192, 16384`. A prefix advances only if its
   JAX-headed full Q is elementwise exact on both corpora.
3. Screen the standalone head on the exact same accepted hidden tensor:
   `BM = 128..2048`, `BK = 128..1024`, `BN = 128`, and both existing Dense
   rounding expressions. Forty Pallas arms are compared with one JAX head.
4. Promote at most three heads that are finite and elementwise exact on both
   legal scrambles and categorical stress.
5. Time the Cartesian product of exact prefix BMs and promoted heads, plus
   one-dispatch Pallas materialization diagnostics at `BM = 128, 512, 2048`.
6. Recompile and confirm only the screen winner and required controls at the
   real local batch 32768.

## Acceptance

- eight active TPU devices, one process, one batch shard per device;
- local batch 16384 for screening and actual local batch 32768 for confirmation;
- identical source, checkpoint, puzzle, corpus seeds, corpus hashes, and shard
  ownership recorded in JSON;
- finite, elementwise-identical full BF16 Q against original `jax_model.apply`
  on both legal scrambles and categorical stress;
- zero mismatch witnesses and exact output hashes;
- paired alternating execution, five warmups and twelve synchronized repeats;
- a new winner must beat the accepted exact split on every corpus at both
  batches; a head-only win is not a full-model win;
- compilation, first execution, HLO, static memory analysis, profiles, and
  warmed execution are reported separately.

Compile rejection has no latency. Profiles are diagnostic and cannot override
the exactness or paired-timing gates. The accepted implementation and public
defaults remain unchanged until a terminal result passes the complete
protocol.
