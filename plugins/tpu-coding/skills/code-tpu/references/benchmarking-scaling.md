# Benchmarking and scaling

Checked: 2026-08-31. This is a measurement protocol, not a fixed winning tile.
See [case studies](case-studies.md) for the shape-specific measurements.

## Make the comparison answer one question

First freeze the checkpoint and executed subgraph, input distribution, batch,
logical/storage shapes, precision, masks, parameter delivery and sharding.
Choose task-quality tolerances before ranking candidate speed. Record rejected
compiles, nonfinite outputs and failed numerical gates separately from accepted
timings; a rejected configuration has no execution latency.

Use two distinct comparisons:

- **Attribution:** identical BM/BK/BN and arithmetic, change only the boundary
  or scheduling decision. Cross Dense/LN implementations independently when
  identifying an operator's contribution.
- **Engineering selection:** each implementation's best passing configuration
  on the actual workload. Do not call a matched-tile experiment a proof that
  every implementation is optimally tuned.

Screen many candidates in one job, save incremental results, then promote a
small set to production batches and full-model/caller tests. Treat tiny
differences against repeat variation, not displayed decimal precision. Keep
JAX as the current reference when it wins the valid full-model comparison;
this does not establish a ceiling on future Pallas work.

## Timing scopes

Measure separately: tracing/lowering/compile; placement/transfers; first
execution; warmed synchronized device execution; full model; actual chunked
caller; complete search depth. Synchronize all relevant output leaves before
stopping the timer, keep inputs device-resident for device-only timing, and
ensure the output depends on the timed work. A repeated constant input tests
that input distribution, not representative frontier quality.

Record samples, warmup/repeat counts, median and spread; interleave A/B order
when drift matters. Separate latency from throughput percentages:
`throughput = useful_states / elapsed_seconds`. Report padded work separately.
Use profiler traces of the composed caller to attribute stalls, transfers or
overlap; allocating two buffers or observing batch amortization is not proof
that all loads are hidden.
[JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html),
[profiling](https://docs.jax.dev/en/latest/201/profiling.html).

## Multi-device claims

For N devices, aggregate throughput `T_N`, and a matching one-device result
`T_1`, report `speedup = T_N/T_1`, `efficiency = T_N/(N*T_1)`.
State whether global work is fixed (strong scaling) or per-device work is
fixed (weak scaling). Distinguish independent shards, communication-inclusive
inference and complete search. Replicated weights and one shard per device do
not measure global selection, deduplication or inter-device transfer costs.

Keep an ordinary sharded JAX reference before custom remote DMA. Validate
collective ordering and completion/lifetimes independently of timing.
[shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html),
[distributed Pallas](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html).

Dense-equivalent FLOP counts are shape-derived conventions. Timing-derived
FLOP/s is not a counter measurement or utilization proof; any peak denominator
needs verified generation, precision, active units and source scope.

## Kaggle and reproducibility

When Kaggle execution is in scope, pin the Git SHA in the launcher and include
it explicitly in JSON alongside runtime/device inventory, checkpoint hash,
input seed/hash/generator, shapes, tiles, flags and timing scope. Keep code in
Git; use the notebook as a reproducible bootstrap, not an unrecorded fork.

Respect the project's active-session policy; TPUBeamSearch uses one TPU session
at a time. Do not restart QUEUED/RUNNING jobs to poll them. Preserve full logs,
partial JSON and terminal status; fix a failed job only after identifying its
reproducible cause. Use configured networking without silently changing VPN or
proxy state. Retain privacy/redistribution review before publishing artifacts.
