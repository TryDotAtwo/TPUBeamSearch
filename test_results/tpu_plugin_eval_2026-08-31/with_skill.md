Fixture-based assessment using `code-tpu` and primary JAX documentation. No TPU execution or new performance measurements were performed.

## A. Residual-stack diagnosis

The hidden tensors are close but unequal. The final-output discrepancy is real for the tested paths, but does not establish amplification caused by Pallas: the reference uses separately compiled blocks, while the hybrid uses a jointly compiled suffix. JIT transformations can change floating-point evaluation. Identical printed BF16/FP32-statistics summaries also do not establish pairwise tensor equality. [JAX numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).

Run one controlled diagnostic job:

- For each tested depth, compute `control[d] = suffix[d](h[d])`. Compare this with segmented and monolithic JAX; compare `suffix[d](p[d])` directly with `control[d]`. Include a zero-replacement route that feeds the identical JAX hidden tensor through the identical suffix.
- At the first discrepant block, cross JAX/Pallas Dense with JAX/Pallas LN independently. Within that harness vary Dense rounding before versus after bias and BF16 versus FP32 LN statistics, keeping weights, inputs, tiles, epsilon, variance estimator and residual/activation order fixed.
- Save direct pairwise differences, mismatch counts, max/mean absolute error and RMSE at Dense, LN, hidden and final outputs; include the actual consumer's selection metrics. Inspect target lowering for the intended casts.

Sources A and B have different rounding contracts: BF16-rounded dot then bias versus FP32 accumulator plus bias then BF16. That is a causal candidate, not an established explanation of these TPU results.

## B. Beam promotion

Do not promote as search-equivalent. This consumer minimizes globally over parent/action pairs; row-wise argmax agreement measures the wrong decision. One parent forward produces all move scores, not separate child forwards. Independent categorical entries are not a representative permutation-state dataset; even arbitrary permutations need not be reachable.

Minimum acceptance measurements:

- Use replay-verified legal scrambles and recorded frontiers at production beam widths, with categorical stress inputs reported separately. Freeze checkpoint, move order, parent order, masks and precision.
- Measure finite unmasked outputs and max/mean absolute error/RMSE. Compare masked row argmin as an auxiliary metric, then flattened global top-K identities, overlap and order. Record best/second-best and K/K+1 score margins, especially near disagreements.
- Test exact ties and near ties. Preserve the consumer's tie ordering: current JAX defaults to stable input order for equal values, so flattened ordering matters; verify the installed version. A tie-aware quality metric must not silently replace deterministic search equivalence. [JAX top_k](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html).
- Apply identical invalid/inverse/padded-action masks. Require zero invalid candidates accepted downstream. With fewer than K valid candidates, top-K can return sentinel slots: retain the valid count and exclude those slots, including the all-invalid case.
- Replay selected moves and compare frontier evolution, solve rate and solution length under equal search budgets. Set tolerances before ranking speed, and time the real caller/search as well as inference.

Passing a finite suite supports the declared acceptance scope, not equivalence over every reachable state.

## C. Memory and alignment

Fix LN first: zero padding preserves a properly initialized Dense reduction, not the LN population. With valid-lane mask `m`, use `mu = sum(where(m,x,0))/130` and, for centered variance, `var = sum(where(m,(x-mu)**2,0))/130`. Match the reference's estimator, statistics dtype, rounding and epsilon. Preserve affine/ReLU order; zero invalid output lanes after affine/ReLU if padding bias makes them nonzero. Initialize Dense input/weight tails to finite neutral values; zero multiplication does not sanitize NaNs.

There is no universal tile or VMEM budget. For the documented older TPU `pallas_call` layout, trailing block dimensions normally obey divisibility by 8 and 128, but may instead equal their corresponding whole-array dimension; width 130 is not automatically illegal. Rank-one, dtype and operation-specific constraints still need checking. [TPU kernel details](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

Budget simultaneously live input/output windows, accumulators, scratch, padding, pipeline buffers and spills. The hardware table lists v3 VMEM as 16 MiB per TensorCore; newer generations differ, and a scoped compiler limit is not total physical inventory. [TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html).

Local `interpret=True` establishes only the tested HLO-interpreted behavior, not target compilation, memory fit, race freedom or speed. Where supported, TPU `InterpretParams` adds memory/DMA/semaphore simulation and optional race detection; it still cannot prove hardware safety or performance. Next gate is target compilation followed by correctness checks before timing. [TPU interpretation](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html).

## D. Performance interpretation

Keep original JAX as the current full-model reference. On the supplied rates it has about 2.53x the per-layer-fused model's throughput. Checkpoint and batch match, but confirm arithmetic, executed outputs, inputs, sharding and timing scope before publishing a controlled comparison.

The 1.5x microbenchmark result belongs to the combined tile/fusion change; it does not isolate fusion. The matched-tile result supports only a measured 2.2% difference, whose significance needs repeated samples and spread. Neither proves a full-model win or that Pallas has reached its ceiling.

Eight devices at 96% efficiency imply 7.68x aggregate throughput versus the matching one-device baseline. Fixed per-device batch makes this weak scaling of independent inference, not strong scaling or complete search scaling. Collectives, global selection, deduplication and search are explicitly unmeasured.

Next: interleave repeated synchronized A/B timings, separate compile/placement/first-call/warmed execution, and synchronize every relevant output leaf. Use matched tiles for attribution and each implementation's best passing configuration for engineering selection. Profile the composed full model, production chunked caller and complete search depth; then measure communication-inclusive scaling and fixed-global-batch strong scaling separately. Count useful parent states, not padded work or move scores as extra forwards. [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html).

## E. Runtime and pipeline design

The log is incomplete inventory. Record Python/JAX/jaxlib/libtpu versions and compiler flags; backend, process index/count, local/global device counts; each device's `id`, `device_kind`, process and exposed coordinates/core information; mesh and array sharding. Establish JAX-device/TensorCore/chip mapping separately. `TFRT_TPU_0`, requested `v3-8` and a scoped 16 MiB limit do not establish those facts.

Do not port CUDA warp/stream scheduling literally. TPU `pallas_call` grids are normally sequential lexicographic loops, except explicitly parallel multicore dimensions. Preserve consecutive updates of each output window; exploit input-window reuse and schedule reduction initialization/finalization correctly. [TPU execution semantics](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

`pl.ANY` is unconstrained placement, not forced HBM. Its data must be copied into an appropriate accessible memory space before use. VMEM/SMEM are software-managed SRAM. Start from the existing pipeline, which already stages transfers; explicit double buffering is an experiment with extra live memory and synchronization obligations. Wait before consuming a DMA destination or reusing its source, destination or semaphore. [TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).

Check installed signatures and flags in the 0.10.2 job; a 0.10.1 CPU result and a latest `pl.kernel` tutorial are not API/compilation guarantees. For example, 0.10.2 changes TPU `needs_layout_passes` to default true. Mosaic-GPU `pallas_call` migration advice is not a universal TPU deprecation. [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html).

Claim overlap only after a correct target run and a profiler trace showing DMA and compute overlapping in the actual caller. Compare matched default/manual schedules with synchronized timings and allocation evidence. Quantify residual transfer stalls and distinguish steady state from pipeline fill/drain; two buffers alone never establish that all loads are hidden.

## F. Constants and experiment maintenance

The historical BN result motivates an A/B, not a diagnosis of this model. Both current wrappers capture weights, so a one-sided-capture explanation is unsupported. Compare a 2x2 matrix: JAX/Pallas by captured/runtime parameters, using identical device-resident weights, numerical contracts and workload. Inspect lowering, constant-handling flags, compile status, scoped allocations and full-caller timings. Runtime parameters may be a useful implementation boundary, but no current speed or memory improvement is established. Constant lowering itself is runtime/flag-sensitive. [JAX constants](https://docs.jax.dev/en/latest/internals/constants.html).

Keep compile-rejected candidates as compile failures with allocation diagnostics, not slow timings. Keep the fastest numerically invalid candidate as a failed quality gate, excluded from deployable winners. Preserve raw results and distinguish documented facts, measurements, hypotheses and superseded interpretations; 44 MiB is not a universal capture limit.

Proposed evidence entry, not a newly measured result:

```json
{
  "id": "H-CURRENT-PARAMETER-DELIVERY",
  "status": "hypothesis",
  "claim": "Runtime-array parameters may reduce current-wrapper allocation pressure; transfer of the historical BN capture fix is untested.",
  "scope": "Scenario F: both current implementations capture weights; exact current hardware, shapes, compiler flags and target A/B artifacts remain unverified.",
  "source_urls": ["https://docs.jax.dev/en/latest/internals/constants.html"],
  "checked_on": "2026-08-31",
  "recheck_when": "A matched target parameter-delivery A/B completes, or runtime, constant-handling flags, shapes or wrappers change.",
  "regression_case": "Cross JAX/Pallas with captured/runtime parameters; preserve checkpoint/input hashes and compare lowering, compile/allocation status, correctness and synchronized full-caller timings. Do not infer a one-sided capture cause or a universal size threshold."
}
```

After actual evidence arrives, attach pinned code/results/logs and complete runtime/input provenance; update only the scoped claim and relevant guide decision. Keep experiment-specific numbers in case studies. Re-run affected regression scenarios, fresh-context application evaluation and package checks; version changed guidance. Expert agreement alone is not a new measurement. No guide mutation or promotion is justified by this hypothetical entry itself.
