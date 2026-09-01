# Final-residual schedule-restoration A/B

## Evidence and target

Exact eight-device execution A/B v1 establishes that Pallas and JAX produce the
same materialized BF16 encoding.  One-device replay on the exact owner shards
reproduces the monolithic Pallas Q drift, while returning internal nodes makes
every observed value exact.  `shard_map`, `pmap` and independent executables do
not change either the oracle or candidate output hashes.  A whole-encoding split
also changes JAX and Pallas to the same third output hash.

Compiled HLO has one changed MXU schedule among the 22 Dense operations: the
second Dense of final residual block 9.  The next run targets only that block.
It does not change model weights, operations, precision, corpus, batch contract
or acceptance gate.

## Frozen arms

All arms use eight TPU devices and local batch 16,384 first.  Controls are the
original `jax_model.apply` and typed-BF16 JAX monolith.  The candidate baseline
is the current BM2048 prepacked Pallas encoding followed by the unchanged JAX
Dense/LN ResMLP.

Five one-dispatch Pallas arms insert exactly one targeted
`optimization_barrier`, except the explicitly named two-barrier arm:

- before final residual block 9;
- before its second Dense;
- after its second Dense and bias, before LayerNorm;
- after the final residual output, before the Q head;
- before and after the second Dense.

Three one-dispatch tap pairs return Q plus one BF16 tensor: input to final
Dense2, output of final Dense2, or output of final block.  Both returned tensors
are synchronized and their write cost is included.  The typed-JAX and Pallas
versions use the same output contract.

Four two-dispatch pairs materialize a device-resident boundary before final
block, before final Dense2, after final Dense2, or after final block.  Tuple cuts
carry both the residual skip and branch tensor.  No intermediate is transferred
to the host.  Typed-JAX and Pallas versions use identical partitions.

Every successful executable saves StableHLO, compiled HLO, compile/first-run
time and compiler static memory estimates.  Compilation or runtime failures are
retained as results rather than silently removed.

## Timing and gate

Five warmups precede twelve paired forward/reverse rounds.  Each timed sample is
synchronized end-to-end runner wall time.  Compilation, first execution, host
conversion, checkpoint conversion and one-time embedding-bank construction are
excluded.  Tap writes and split dispatch overhead are included.

The oracle is the full BF16 Q tensor from original `jax_model.apply` under the
canonical eight-device `shard_map`.  An arm is eligible only if it is finite and
elementwise exact on both legal scrambles and categorical stress.  It must beat
the fastest exact JAX control separately on both corpora at local batch 16,384.
Only a winning arm, the canonical controls, the measured baseline controls and
its structurally paired JAX arm advance to an actual local-batch-32,768 run.
The same candidate must remain exact and faster on both corpora there.

Argmin agreement, maximum/mean absolute error and output hashes remain
diagnostics; none can override exact Q.  This is inference only: no expansion,
top-k, deduplication, parent bookkeeping or other beam-search stage is present.
