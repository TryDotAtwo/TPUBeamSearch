# Exact eight-device inference optimization

This experiment optimizes only the complete Artgor Cube555 Q-network forward.
Move expansion, global top-k, deduplication, parent tracking and all other beam
search stages are outside the executable and outside every timer.

## Hypothesis

The accepted one-device hybrid spends about0.666ms per16K call constructing
phase-indexed Pallas lookup banks from the unchanged150x24 embedding. Those
banks depend only on the checkpoint, so a deployment can construct them once,
synchronize them, and pass them as ordinary replicated runtime model arrays.
The residual ResMLP, all21 LayerNorm operations and the30-score head stay in
JAX/XLA because the measured Pallas replacements were slower and failed the
unchanged exact-Q gate.

The new sweep crosses BM64/128/256/512/1024/2048 with BF16 and FP32 physical
bank storage. Both storage modes first round the checkpoint embedding to BF16;
the axis changes storage and transfer cost, not logical lookup values. The
existing per-call FP32-bank implementation and JAX tiled lookup remain controls.

## Promotion protocol

- Same checkpoint, model source, puzzle and deterministic legal/stress corpora.
- Original JAX, typed JAX and candidates all receive replicated runtime arrays.
- One `shard_map` dispatch shards only batch axis; every device receives the
  same local batch and an independent contiguous input shard.
- All executables are lowered, compiled and executed once before timing.
- Five warmups and12 synchronized timed rounds alternate forward/reverse order.
- Compilation, model prepacking, host transfer, hashes and quality analysis are
  excluded from steady inference timings.
- Full BF16 Q arrays are compared elementwise against original `jax_model.apply`.
  A candidate must be finite and exact on both legal and categorical-stress
  inputs. Lower-Q argmin and ranking metrics are retained for the final winner.
- The eight-device winner must beat original JAX separately on both corpora at
  local batch16,384. The same candidate must repeat that result at local
  batch32,768; the gate is not based on an average that can hide one loss.
- Weak scaling uses fixed local work: global batch is `devices * local_batch`.
  Throughput is global states/second; efficiency is 1-to-8 throughput speedup/8.

The job records actual TPU device inventory, source/checkpoint/model/puzzle and
input hashes, per-shard input hashes, bank layouts/hashes, all raw timing samples,
static compiler memory reports, StableHLO, compiled HLO and diagnostic profiles.
Static allocation reports are not hardware bandwidth or utilization counters.
