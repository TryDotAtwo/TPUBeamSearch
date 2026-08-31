# Arithmetic follow-up bundle

Approved and implemented 2026-08-31. This is an experiment protocol, not a
TPU success or performance report. It follows the
[HLO/compiler investigation](2026-08-31-arithmetic-followup-research.md).

## Fixed model and validation

Artgor cube555: state150, embedding150x24, flattened3600, hidden1024,
ten two-layer residual blocks, epsilon1e-5, Q30 (minimize).
All full candidates retain the JAX embedding/input layer and Q head; the
residual Dense/LN implementation is the only replacement boundary. A full
forward does not mean an all-Pallas model.

Use the original runtime-FP32-weight JAX implementation as the exact-output
oracle and the equivalent runtime-BF16-weight JAX model for attribution.
Include a full-builder JAX/JAX control and a separately named captured-weight
control. Preserve v1 legal32768 seed42 and categorical-stress seed43 arrays;
compare the recorded checkpoint/source/puzzle/input hashes against v1.

Only finite, elementwise-exact Q on **both** 16K corpora qualifies a non-control
candidate. At most two qualified candidates proceed to actual32K measurement;
qualification is not32K confirmation. No relaxed tolerances, argmax-based
promotion or inferred distributed-beam validity. Existing minimizing top-K,
tie/margin/order diagnostics and inverse-mask diagnostics remain available.
The recorded primary configuration allows immediate inverse moves.

## One queued job

The new entrypoint is `benchmarks.stream1_layernorm_followup`.

1. **56 synthetic cases:** BM128/256, logical widths1024/130; minimal
   alternating predicates with BF16/FP32 operands and rank1-broadcast/direct2D
   predicate construction; legacy BF16 versus HLO-informed mixed LN with
   mask-site isolation. Width130 retains all population/output masks.
2. **36 checkpoint operator cases:** on the same block1 JAX hidden/Dense input
   per corpus, standalone JAX/late Dense, original JAX/v1 LN, and fourteen
   arithmetic/mask arms. Compiled Dense and LN groups have matched JAX controls.
3. **14 block cases:** seven configurations on each corpus at4096. Every
   candidate and the JAX control feeds the identical compiled JAX suffix.
   Separately retain same-suffix and monolithic-Q comparisons.
4. **14 full cases plus6 baselines:** seven configurations on both16K corpora,
   with original-runtime, typed-runtime and captured JAX baselines each.
5. **Conditional32K:** at most two exact non-control candidates, both corpora.

Full configurations:

| ID | Residual Dense | Residual LN |
|---|---|---|
| jax-graph-control | JAX | JAX |
| late-dense-jax-ln | Pallas late rounding | JAX |
| jax-dense-legacy-unmasked | JAX | BF16, redundant masks removed |
| jax-dense-legacy-fp32-where | JAX | BF16, selects promoted then converted back |
| late-dense-legacy-unmasked | Pallas late | BF16, redundant masks removed |
| jax-dense-mixed | JAX | HLO-informed mixed arithmetic |
| late-dense-mixed | Pallas late | HLO-informed mixed arithmetic |

`none/input/center/output` mask modes are rejected whenever logical width
differs from aligned storage width. Row padding is sliced away as in v1;
there is no additional row predicate contaminating the three-site experiment.
Direct2D source construction may still be canonicalized by the target
compiler. Mixed `all` and `fp32_where` have the same arithmetic because their
select inputs are already FP32; do not count these as independent numerical
approaches.

## Timing and artifacts

- Pin JAX/jaxlib0.10.2 and libtpu0.0.42.1; inventory actual hardware. The prior
  runtime was TPU v5 lite/v5e despite a requested `v3-8` accelerator.
- Save StableHLO before compilation and compiled HLO afterward. Compile
  rejections have no execution latency; keep full tracebacks per case.
- Five warmups and twelve synchronous measured rounds, alternating
  forward/reverse case ordering, with runtime-resident arguments.
- Eight retained calls to the **same executable**, five timed queue batches,
  as a separate amortized-throughput diagnostic for operators/blocks/full
  models. This is neither one-call latency nor a real128-chunk scan.
- Profiles of compiled16K legal baselines/candidates after timing, three calls
  each, regardless of Q acceptance. Failed quality never becomes a speedup
  claim merely because a trace exists.
- Incremental strict JSON, compile/error logs, paired sample ordering, queued
  samples, profile errors and exact-Q status. A timing-group failure triggers
  explicitly unpaired diagnostic fallback; that group cannot promote winners.
- Count actual unique states and `(state,last_move)` pairs at16K without
  replacing or deduplicating the original acceptance corpus.

## Scope exclusions

No production BN/LN defaults, dataset publication, compiler-switch bypass,
multi-device scaling, new model training, beam-selection implementation or
acceptance-threshold changes. The new experimental module is opt-in. CPU
tests verify expression/protocol behavior only; TPU compile/numerical results
must come from the launched job.
