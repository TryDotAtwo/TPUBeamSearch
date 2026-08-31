I read only the scenario fixture and public primary JAX documentation. No files were changed and no TPU jobs were run.

## A. Residual-stack diagnosis

The result establishes a discrepancy in this diagnostic, not its cause. High hidden cosine does not imply preserved output decisions: the reported depth-1 substitution has only 73.01% final argmax agreement.

The comparison has an important confound: the reference executes separately compiled blocks, whereas `suffix[d]` compiles several blocks together. Thus `suffix[d](h[d])` itself may differ from the stated reference. JIT compilation can change floating-point evaluation; the Pallas contribution cannot be isolated without that JAX-only control. [JAX FAQ](https://docs.jax.dev/en/latest/faq.html)

Identical printed max/mean/RMSE summaries do not establish elementwise equality between the BF16-statistics and FP32-statistics variants, nor prove that statistics precision never matters.

Dense A and B also have potentially different rounding contracts: a BF16 matmul result followed by BF16 bias addition versus bias addition before the final BF16 conversion. BF16 operands/results do not imply that the physical matmul accumulator is BF16. Inspect actual lowering instead of attributing the discrepancy to accumulator precision from source syntax alone.

Next, run one compact controlled diagnostic:

- Freeze checkpoint, valid-state batch, tiles, precision settings, and the intended production JAX reference.
- For each sampled depth—at least 0, 1, an intermediate depth, and 10—evaluate the same compiled suffix on `h[d]` and `p[d]`. Report both `suffix[d](h[d]) − y_staged` and `suffix[d](p[d]) − suffix[d](h[d])`; also compare the staged and production-monolithic JAX outputs.
- At the first divergent block, feed the identical JAX hidden input into JAX and Pallas. Cross two Dense rounding schedules with BF16/FP32 LayerNorm statistics, keeping everything else fixed.
- Capture post-Dense, statistics, affine/ReLU, and residual-add outputs; report exact unequal-element counts, max/mean/RMSE, and ReLU branch disagreements. Compare the two statistics variants directly.
- Inspect lowered and optimized IR for casts, bias placement, reductions, epsilon, and residual ordering. Then evaluate output decisions, including the actual beam selection from B.

This separates compilation-boundary drift, local arithmetic differences, and downstream sensitivity without prematurely committing to a larger rewrite.

## B. Beam promotion

Do not declare search equivalence or promote on this evidence.

The search selects the globally smallest scores across all parent–move pairs. Row-wise **argmax** agreement measures neither the correct direction nor the global beam cutoff. Even perfect per-row ordering would not ensure cross-parent score calibration or identical global selection.

Independent integer draws are not permutation states. Use replay-validated states obtained through legal moves and recorded production frontiers; a randomly shuffled permutation is still insufficient if the puzzle imposes additional reachability constraints.

Minimum acceptance measurements:

- Compare full-model scores on representative valid frontiers, including tail batches, depths, masks, and difficult near-cutoff cases. Report finite counts and score-error distributions.
- Apply the identical invalid-move mask and flattening convention; compare selected flat IDs, decoded parent/move IDs, selected-set overlap, ordered-ID equality, and cutoff disagreement at production beam widths.
- Measure the `k` versus `k+1` score gap and perturbations around that boundary. A small average error is not a cutoff guarantee.
- Test exact ties and BF16-created ties explicitly. In JAX 0.10.2, equal `top_k` entries choose the lower index first; preserve this flattened-index rule. Tie-aware set overlap is useful diagnostic information, but does not replace exact ordered agreement when the deployed search depends on deterministic IDs. [JAX 0.10.2 `top_k` source](https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/jax/_src/lax/lax.py)
- Test partial masks, fully masked rows, all-masked frontiers, and fewer than `beam_width` valid candidates. Invalid `+inf` becomes `-inf` after negation; `top_k` still fills its requested count, so downstream validity handling must prevent invalid filler entries from becoming real children. Check NaN/nonfinite handling separately.
- Replay complete fixed-seed searches, including downstream deduplication and pruning. Measure frontier divergence, solve rate, solution lengths, replay validity, and end-to-end time under matched budgets.

For an exact-equivalence claim, require exact selection behavior on the defined acceptance suite and explain that finite testing is not a universal proof. If approximation is acceptable, agree explicit search-quality tolerances and call it an approximate replacement—not search-equivalent.

## C. Memory and alignment

Zero padding does not make LayerNorm over 256 equivalent to LayerNorm over 130.

Let `r = 130/256`, and let the logical values have mean `μ` and variance `σ²`. Adding zero-valued lanes gives:

`μ_padded = r μ`

`σ²_padded = r σ² + r(1−r) μ²`

Both generally differ from the logical statistics. Slicing afterward cannot undo the changed normalization.

Compute masked statistics over exactly 130 lanes, dividing by 130. For a centered-variance implementation, mask `(x − μ)²` as well: padded lanes otherwise contribute `μ²`. Match the reference reduction algorithm, statistics dtype, epsilon placement, affine calculation, and output rounding. Ensure Dense reduction padding is neutral and padded outputs remain neutral before later operations; padded affine parameters alone do not guarantee this.

For the older `pallas_call` API, use its version-specific constraints: TPU block shapes have rank at least one; the final two dimensions generally must be divisible by 8 and 128 respectively, **or equal the corresponding full-array dimensions**. Narrower-dtype accesses and operations have additional layout/slicing restrictions. This is an admissibility rule, not one universally optimal tile. [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)

Budget actual padded input/output tiles, all buffer copies, FP32 accumulators, temporaries, scratch, and compiler spills. Determine physical generation and the effective compiler limit separately. VMEM capacities vary across generations; a single universal v3-and-newer budget is inappropriate. [TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)

`interpret=True` passing establishes only that the interpreted execution satisfies the tests that were actually run. If its oracle also normalizes 256 lanes, it validates the wrong semantics. It does not establish TPU compilation support, hardware numerical agreement, VMEM fit, absence of hardware-specific synchronization problems, or performance. Before a real run, use a logical-width oracle, padding/adversarial cases, and version-matched lowering checks; hardware compilation and execution remain separate gates.

## D. Performance interpretation

Keep the original JAX model as the current reference: 1.386 versus 0.547 M states/s. In this fixture the fused model delivers about 39.5% of JAX throughput; JAX is approximately 2.53× faster.

Supported claims:

- The complete changed microbenchmark configuration is 1.5× faster. That result cannot be attributed to fusion because BM/BK/BN changed simultaneously.
- The matched-tile observation is a 2.2% improvement, subject to repeatability and measurement uncertainty.
- The tested per-layer-fused full model is substantially slower than the matched JAX model.
- Eight independent shards show 96% **weak-scaling** efficiency for the timed workload. Under the conventional throughput definition, that corresponds to approximately 7.68× aggregate throughput relative to one device.

Unsupported claims include fast global beam search, cheap collectives, strong scaling, or a fusion-induced 1.5× gain.

Next measure:

1. A controlled tile/fusion ablation, with compilation and placement excluded from warm execution timings.
2. Repeated synchronized full-model timings and a profile locating extra launches, materialized intermediates, transfers, conversions, or underutilized kernels.
3. Complete search-step time: expansion, inference, masking, top-k, deduplication, state movement, and any communication.
4. Both fixed-global-work strong scaling and fixed-per-device-work weak scaling, explicitly labeling timer boundaries.

Use device-ready inputs, warmup, and completion synchronization; asynchronous dispatch can otherwise make host timings misleading. [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)

## E. Runtime and pipeline design

The log is not complete hardware inventory.

`TFRT_TPU_0` is a device identifier, `v3-8` is the requested allocation, and `16.00 MiB` is an observed compiler-scoped VMEM limit. Record actual device kind, local/global device counts, process count/index, chip/core mapping and topology where available, addressable shard placement, memory information, and the effective compiler configuration.

Record JAX, jaxlib, libtpu/plugin versions and relevant flags from the **job**. Local 0.10.1 interpretation does not validate 0.10.2 TPU compilation.

Do not copy CUDA warp/stream scheduling literally. TPU Pallas work is organized around vector/matrix operations, grid/core mapping, and DMA pipelines, not CUDA warps. [Pallas programming model](https://docs.jax.dev/en/latest/pallas/quickstart.html)

`pl.ANY` leaves placement unconstrained; it usually means HBM but does not force HBM. Its buffers require appropriate copies into compute-accessible memory. Use the target version's supported explicit placement mechanisms when placement is a requirement, and verify the resulting lowering. [TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)

Do not migrate APIs just because a current tutorial uses `pl.kernel`. The tagged JAX 0.10.2 source exports **both** `kernel` and `pallas_call`. Match examples, imports, signatures, memory semantics, and compiler parameters to the pinned runtime; test migration separately. [JAX 0.10.2 Pallas exports](https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/jax/experimental/pallas/__init__.py)

Two allocated buffers alone do not establish overlap. A valid pipeline needs future-copy starts before current compute, waits at the correct consumption/reuse points, no live-buffer overwrite, and correct prologue/epilogue behavior. Existing pipeline machinery may already allocate multiple buffers; adding another layer can only increase VMEM pressure without helping. [Software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)

Before claiming overlap, require a hardware profile or compiler schedule demonstrating DMA concurrent with useful compute, plus a synchronized matched comparison against a non-overlapped implementation. Examine DMA wait stalls, compute utilization, HBM traffic, and tile-count dependence. “All loads hidden” is stronger still: startup/drain costs and residual stalls must be accounted for. Without such evidence, describe overlap as intended, not measured.

## F. Constants and experiment maintenance

The historical result makes explicit runtime weight arguments a sensible controlled experiment, not a diagnosis of the current A/B.

Both current wrappers capture weights, so “Pallas captures constants” cannot by itself explain their difference. The implementations may nevertheless lower those closures differently. Closed-over constants can affect HLO size, compilation, placement, and numerical behavior depending on JAX's constant-handling configuration. [JAX constant-handling notes](https://docs.jax.dev/en/latest/internals/constants.html)

Next implementation experiment: compare both implementations with explicit device-resident weight arguments, retaining closure-based controls. Keep checkpoint, input, tiles, compilation settings, donation/sharding, and timing scope fixed. Inspect whether large weights become runtime parameters or embedded constants. Do not introduce repeated host-to-device weight transfers into the execution benchmark.

Handle the other candidates independently:

- VMEM-rejected candidate: retain the rejection and compiler diagnostics; no execution throughput exists for that configuration. Reduce live padded tiles/buffering/scratch only as a separately tested change.
- Numerically invalid fastest candidate: retain its raw timing as diagnostic data, mark it ineligible for the valid-performance ranking, and do not promote it.
- Preserve both failures in the reusable guide with exact scope and reproduction metadata; do not silently remove unsuccessful sweep points.

One proposed evidence entry, explicitly not a new measurement:

```yaml
id: constants-history-fixture-2026-08-31
evidence_kind: synthetic_fixture_reporting_historical_observation
source: docs/research/tpu-plugin-eval-scenarios.md#f
scope: historical different-BN experiment
observation:
  captured_weight_size: 44 MiB
  captured_weights_result: failed
  explicit_runtime_arrays_result: passed
unknown:
  - exact hardware and software versions
  - original failure diagnostics and raw artifact locations
  - applicability to the current A/B
next_test: matched implementation-by-weight-passing 2x2 comparison
promotion_status: not established
```

This does **not** establish a universal 44 MiB constant limit, prove the current VMEM rejection has the same cause, show that runtime arguments eliminate all VMEM pressure, or demonstrate any speed/quality benefit for the next implementation.
