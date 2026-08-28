# Universal BN/LN Stream1 Design

## Goal

Extend the existing Stream1 inference engine so it supports both the current
folded-BatchNorm MLP and Andrey Lukyanenko's embedding-based LayerNorm ResMLP,
without regressing the measured BatchNorm path. Determine the best categorical
input encoding experimentally rather than selecting it in advance.

## Supported model families

The first LayerNorm target is the Cube555 ResMLP family:

```text
categorical state [B, STATE_LEN]
embedding [NUM_CLASSES, EMBED_DIM]
flatten
Linear(STATE_LEN * EMBED_DIM, HIDDEN)
LayerNorm(HIDDEN) + ReLU
RESIDUAL_COUNT * {
  Linear(HIDDEN, HIDDEN) + LayerNorm + ReLU
  Linear(HIDDEN, HIDDEN) + LayerNorm
  skip + ReLU
}
Linear(HIDDEN, MOVE_COUNT)
```

The initial concrete checkpoint has `STATE_LEN=150`, `NUM_CLASSES=150`,
`EMBED_DIM=24`, `HIDDEN=1024`, `RESIDUAL_COUNT=10`, and `MOVE_COUNT=30`.
These values are configuration, not kernel constants.

The existing folded-BatchNorm model and its public interfaces remain supported.

## Universal engine

The public engine uses one architecture contract with a static normalization
mode (`FOLDED_BATCH_NORM` or `LAYER_NORM`) and a static input encoding plan.
JAX specializes and compiles each combination; kernels contain no per-state
runtime branch between BN and LN.

Shared responsibilities are architecture validation, alignment, JIT creation,
dense and residual execution, correctness gates, and benchmark reporting.
Normalization-specific weights and execution remain separate internally:

- BatchNorm is folded into dense weights before inference.
- LayerNorm retains scale and bias and computes per-row mean, variance, and
  reciprocal square root at inference time. Reductions and normalization math
  use FP32; dense storage and layer boundaries use BF16 unless an experiment
  explicitly states otherwise.

## Input encoding candidates

No candidate is the selected implementation until TPU measurements complete.

1. `EMBEDDING_GATHER`: reproduce Andrey's original lookup, flatten, and dense.
2. `VIRTUAL_ONE_HOT_MXU`: represent each categorical state value as a virtual
   one-hot row and use MXU-friendly multiplication. The dense one-hot tensor is
   never materialized in HBM.
3. `FUSED_VIRTUAL_ONE_HOT`: fuse categorical expansion, embedding projection,
   and the first dense stage when legal, avoiding the `[B, STATE_LEN*EMBED_DIM]`
   intermediate.

All three consume the same embedding table and first-layer weights and must be
mathematically equivalent within the declared BF16 tolerance.

## Measurement protocol

First measure the unmodified `jax_model.apply` baseline. Then compare every
candidate under identical conditions:

- identical checkpoint tensors and generated categorical states;
- identical local and global batch sizes;
- identical TPU device count and sharding;
- compilation excluded from timing;
- fixed warmup and timed iteration counts;
- `block_until_ready()` at timing boundaries;
- separate measurements for input-prefix only and complete MLP;
- report latency, states/s, speedup versus original JAX, and eight-device
  parallel efficiency;
- retain raw JSON and logs for reproducibility.

Primary production selection uses complete-network states/s at the real search
batch/chunk shape. Prefix-only timing diagnoses the reason for the result but
cannot select a slower full-network path. If results overlap within run noise,
prefer the simpler implementation and record that the performance difference
was inconclusive.

## Correctness gates

The original JAX implementation is the semantic oracle. Before accepting any
performance result:

- imported weights and architecture shapes must validate;
- our JAX reference must match the original implementation;
- every Pallas candidate must produce finite outputs;
- report max absolute and mean absolute error against the oracle;
- compare output argmax agreement;
- run deterministic edge states as well as seeded random states;
- existing BatchNorm tests and benchmark contracts must continue to pass.

Numerical thresholds are established from the reference-vs-Pallas BF16 error
distribution and written explicitly into tests before declaring a candidate
valid; they are not loosened merely to admit a faster kernel.

## Optimization order

1. Import and reproduce the original model and benchmark.
2. Add the universal architecture/weight contract while preserving BN behavior.
3. Implement the LayerNorm JAX reference.
4. Benchmark the three input-prefix candidates.
5. Implement and tile Pallas LayerNorm.
6. Optimize one residual block, then the full ten-block stack.
7. Benchmark full inference on 1 and 8 TPU devices at microbenchmark and real
   scan chunk shapes.
8. Select and document the production encoding and tiling only from successful
   correctness-gated measurements.

## Non-goals

- A completely arbitrary neural-network graph.
- Training or backward propagation.
- Replacing the existing BN model or changing its selected tiling without a
  separate measured regression study.
- Physically allocating full one-hot input tensors in HBM.
