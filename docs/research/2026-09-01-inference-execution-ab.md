# Exact eight-TPU inference execution A/B

## Why this run exists

The first eight-device run found a fast physical embedding path but did not
promote it.  At local batch16,384, prepacked Pallas BM2048 reached17.41 and
17.92 million states/s across eight TPU v5 lite devices, versus10.77 and10.74
million states/s for original JAX.  It differed from the oracle in12 of
3,932,160 legal Q values and55 of3,932,160 stress Q values.  All Pallas tile
sizes, runtime banking and tiled JAX produced the same candidate output hash.

The original one-device screen saw only the first16,384 inputs.  Its exactness
therefore does not prove that device count caused the drift: the known legal
witnesses are global rows29,807 and50,224, and the first retained stress
witness is29,369.  This bundle first replays the exact owner shards on one TPU.

## Frozen localization

For each known witness, record the global row, owner core, local row, state
bytes and hash.  Replay the whole16,384-row owner shard with original JAX,
typed JAX and the current Pallas BM2048 model.  Separately observe the witness
row at encoded input, input Dense, input LayerNorm/ReLU, every residual block
and final Q.  Observed-node graphs are diagnostic because returning internal
values may change fusion; the unobserved full replay remains the correctness
evidence.

## Frozen execution matrix

All arms keep the checkpoint, BF16 values, ten residual blocks and30-output Q
head unchanged.  The matrix contains:

- original and typed JAX through the existing `shard_map`;
- monolithic Pallas lookup with no input boundary and with pre/post/both
  `optimization_barrier` placements around the first Dense;
- explicit two-dispatch JAX and Pallas encoding-to-MLP pipelines, whose BF16
  intermediate stays device-resident;
- direct `jit` with explicit global in/out shardings;
- `pmap` with a leading replica axis and preplaced per-replica weights;
- eight independent one-partition AOT executables, launched asynchronously in
  device order and synchronized once at the end.

The last mode is labeled separately because it changes host orchestration.  Its
wall time includes all eight enqueue calls and final synchronization.  No host
output transfer, compilation, first execution, model conversion or bank
preparation is timed in any arm.

## Gate

The canonical oracle is original `jax_model.apply` under the existing
eight-device `shard_map`.  Every result is flattened in contiguous shard order
and compared against that full BF16 Q array.  A candidate must be finite and
elementwise exact on both legal scrambles and categorical stress.  At local
batch16,384 it must beat the fastest exact JAX control separately on each
corpus, then the same winner must repeat the gate at local batch32,768.

Compilation failure, hidden resharding, non-comparable timing or one failed
corpus rejects an arm.  Argmin agreement is retained as an additional metric
but cannot replace exact Q.  The run contains no move expansion, top-k,
deduplication, parent tracking or other beam-search work.
