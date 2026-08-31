# TPU coding research and experiment audit

Date: 2026-08-31. Repository examined at `fb84b7c`. This is research for the
proposed `tpu-coding` plugin, not an installed plugin or a new TPU result.
No inference implementation or Kaggle session was changed during this audit.

## Evidence and consultation scope

- Reviewed the supplied conversation, the repository history, 22 published
  experiment reports from August 28–29, their benchmark sources, and relevant
  JSON results. The supplied conversation is not a complete export of every
  prior assistant/tool turn; repository artifacts provide the executable evidence.
- Consulted `multigpu_beam` and `multigpu_mlp` in one shared expert request.
  The detailed project-data request was blocked by the safety reviewer. A second,
  generic methodology request without checkpoint or benchmark details succeeded.
  Their recommendations are peer advice, not TPU measurements or hardware authority.
- Independently checked official JAX/Cloud TPU sources below. Documentation is
  a moving target: our historical runs recorded JAX 0.10.2; local CPU checks here
  used JAX 0.10.1. Current documentation also describes newer/unreleased APIs.

Both experts emphasized three distinct contracts: tensor/model semantics,
numerical semantics, and measurement scope. They recommended intermediate
comparisons, interpretation before hardware execution, explicit padding/masks,
ranking-aware checks, synchronized timing, and full-pipeline promotion gates.
These recommendations fit our code audit. A general expert claim that TPU loops
are fully unrolled was not verified in the current TPU-details page and is not
adopted as a rule. Interpreter behavior and supported operations are version-specific.

## Corrections to the previous conclusions

### 1. The depth experiment is missing a JAX-only boundary control

In `benchmarks/stream1_layernorm_depth_diagnostic.py`, `reference_hiddens` is
built by executing individually jitted blocks (lines 188–192). Each suffix is
instead compiled as one function containing all remaining blocks (194–201).
The hybrid is compared to the separately executed oracle (281–293), but the
following control was never recorded:

```python
suffix_calls[depth](reference_hiddens[depth])
```

Compare that JAX-only suffix output to both the segmented oracle and original
whole-model `jax_model.apply`. Then compare the Pallas hybrid to the same-suffix
JAX control. JIT boundaries can change floating-point evaluation, as the
[JAX FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)
explicitly documents.

The recorded 0.000545 mean hidden error and 73.01% final argmax agreement at
depth 1 remain measurements of this harness. They do **not** establish that
Pallas error was amplified by the network. The control gap is confirmed; its
numerical contribution on TPU is not yet measured.

### 2. Argmax agreement is not this Q-beam's action-selection agreement

The benchmark records row-wise `argmax`, e.g.
`benchmarks/stream1_layernorm_full_mlp.py:146`. The inspected Artgor engine
selects smaller Q values: `_topk_smallest` uses `top_k(-values)` at lines 157–159,
and the Q path masks inverse moves before smallest-score selection (675–698).

Consequently, the old description "74% agreement of the chosen move" is wrong.
The actual measurement is approximately 74% agreement of the row maximum.
It remains a numerical diagnostic, but it is not the solver's quality metric.
This correction does not make any Pallas candidate correct or faster.

The next gate needs row-wise **argmin**, score direction in the contract,
identical inverse-move/validity masks, and overlap/order of the actual flattened
candidate top-K. Add best/second-best and K/K+1 margins and tie handling. Do not
replace the existing 99% argmax gate with an invented universal argmin threshold:
choose acceptance against the intended search quality and replay requirements.

Provenance of inspected, existing untracked external artifact:
`test_results/artgor_cube555_tpu/jax_beam_spmd_v_only.py`, SHA256
`aaa0dbe16fd82a0f2bc08f1216f4e87c8a2a63c855f5d7012b6c18d8b57d42cb`.
Original notebook: [Artgor cube555 TPU beam](https://www.kaggle.com/code/artgor/cayleypy-cube555-tpu-beam-q/notebook?scriptVersionId=344319112).
This audit does not stage or redistribute that external artifact.

### 3. Dense rounding points differ at the expression level

The reference uses BF16 `x @ w + b`; its Jaxpr has a BF16 dot result followed
by a BF16 add. Our Pallas Dense adds FP32 bias to the FP32 accumulator before
the final BF16 conversion:

- Reference: depth diagnostic lines 153, 163, 171, 182.
- Pallas: `src/tpu_beam_search/stream1_layernorm_pallas.py:158` and `:378`.
- Shared Dense: `src/tpu_beam_search/stream1_pallas.py:307`.

An independently rerun local CPU witness uses `np.random.default_rng(4)` and
three consecutive `normal(size=...)` draws for x, w, b, shaped `(4,8)`, `(8,8)`,
`(8,)`, each cast with `jnp.asarray(..., dtype=jnp.bfloat16)`. It gives max
absolute difference **0.03125** between jitted
`x @ w + b` and jitted `(fp32(x) @ fp32(w) + fp32(b)).astype(bf16)`.

This demonstrates a real difference between the expression contracts. It does
not measure compiled TPU rounding, the checkpoint's error, or the share of the
observed discrepancy explained by bias. Inspect lowered code and A/B explicit
rounding boundaries on the target runtime before assigning causality.

### 4. Equal aggregate metrics do not prove identical tensors

The depth JSON saves max/mean error, RMSE, cosine, exact fraction against an oracle,
and argmax agreement. Equal summaries for per-layer and per-block variants do
not establish pairwise element equality. Previous "identical tensors" wording
was too strong. Add direct pairwise equality and maximum pairwise error; use
hashes only as reproducibility aids, not a substitute for a direct comparison.

### 5. In-domain categorical inputs are not necessarily reachable puzzle states

`make_valid_states` in `benchmarks/stream1_layernorm_full_mlp.py:24` hashes row
and column indices into `0..149`. It tests categorical coverage but does not
preserve the picture cube's permutation/reachability constraints. Keep such
stress inputs, label them correctly, and add legal-move scrambles and saved
beam frontiers for task-quality decisions. Repeated identity rows are another
distinct, degenerate dataset, not a substitute for representative traffic.

Independent local check: `make_valid_states(64,150,150)` has 86–104 distinct
classes per row, and zero of the 64 rows are permutations of all 150 classes.

## Contracts recovered from the conversation and code

| Contract | BN categorical MLP | Artgor LN embedding MLP |
|---|---|---|
| Logical state / stored state | 120 /128 | 150 /150 in the measured harness |
| Classes / encoding width | 120 /14400 virtual one-hot | 150 /24 embedding, flattened3600 |
| Hidden graph | 1536→512, two residual blocks at512 | input1024, ten residual blocks at1024 |
| Requested output | 24 move scores per parent | 30 move scores per parent |
| Normalization | inference BatchNorm folded into affine weights | per-row runtime LayerNorm,21 norms total |

The earlier capability smoke's scalar-output 2048→512/eight-residual model is
not either of these comparison workloads. An early "full MLP" report omitted
residual blocks; keep its timings labeled as the actual subgraph.

The user required a central architecture definition, constants such as
`MOVE_COUNT`, and separation of logical sizes from padded storage/tile sizes.
For LN this also requires separate logical normalization width: zero padding is
neutral for a dense reduction but not for the denominator of an ordinary mean.
The requested programming stack is Python+JAX+Pallas, with hardware-aware
stage boundaries; a whole beam depth need not become one Pallas kernel.

Historical design intent is not measured success. `RESEARCH.md` is an initial
proposal, not a completed beam-search implementation or final selection strategy.
For an exact hierarchical top-K without later filtering, union of each shard's
local K contains global K under one common total ordering; local K/device_count
does not have this guarantee. Deduplication or other later rejection changes
the completeness argument, so arbitrary oversampling is not an exactness proof.

## What the experiment history actually supports

| Observation | Evidence | Scope and reusable lesson |
|---|---|---|
| BN complete model reached 25.482M states/s on 8 independently sharded devices, 96.04% weak-scaling efficiency | `test_results/kaggle_stream1_optimized_scaling_2026-08-28.md` | Fixed local batch 32768, replicated weights, no collectives. Not a beam-search scaling result. |
| Enlarging prefix tiles improved the BN full model to 3.313M states/s per device | `test_results/kaggle_stream1_prefix_optimization_2026-08-28.md` | Runtime weights, BM1024/BK128/BN1536, local batch32768. Recheck other shapes and hardware. |
| Manual one/two-buffer/lookahead schedules did not improve the tested default schedule | Same prefix report | No evidence that missing manual double buffering caused that bottleneck; overlap itself was not established by a profiler trace. |
| BN prefix versions 1/2 captured weights and hit a scoped allocation failure; runtime arguments fixed it | Same report, commit `976072d` | Test the real caller and parameter delivery, not only a kernel body. |
| LN embedding gather beat virtual one-hot MXU in the tested prefix | `test_results/kaggle_layernorm_input_ab_2026-08-28.md` | BN direct categorical input and LN learned embedding are different workloads. Pre-folded embedding variant was compile-rejected, not proven slow. |
| LN full model measured JAX1.386M versus Pallas0.547–0.577M states/s | `test_results/kaggle_layernorm_comprehensive_2026-08-29.md` | Historical synthetic-input result; not a ceiling on Pallas. Numerical/task gates need the corrections above. |
| 8/32 block candidates exceeded a 16MiB scoped VMEM limit | Same comprehensive report | Record compiler allocation limit separately from physical hardware VMEM. Include scratch, buffers, spills and lifetimes. |
| Fastest small-batch block was not fastest at batch32768 | Same comprehensive report | Screen cheaply, then promote and remeasure production shapes. Fusion and tile changes need separate A/B axes. |

Additional corrections for future summaries:

- The LN trunk has **21 LayerNorms**: one input norm plus two in each of ten
  residual blocks. The original checkpoint parameter count also includes an
  auxiliary value head, whereas Q-only inference does not execute that head.
- BN logical dense-equivalent work is 47,931,392 FLOP/state; LN Q-path dense work
  is 49,377,280 FLOP/state. Neither counts every actual instruction, padding,
  normalization, encoding cost, or hardware utilization. Report the formula.
- BN 25.482M states/s ×24 is 611.6M output scores/s, not independently evaluated
  child states/s. Q inference evaluates parents once; scalar-V inference evaluates
  children and is a different workload.
- The LN input timing 11.166ms versus8.369ms means about33.4% higher latency or
  25.1% lower throughput, not the same percentage for both denominators.
- LN full-model comparisons capture weights on **both** sides. There is no
  confirmed one-sided constant/dynamic mismatch; compare both with runtime
  parameter arguments for the future reusable API.
- Older prose labels runs v3/v5e inconsistently. Relevant JSON has `str(device)`
  but not `device_kind`. Do not infer physical generation solely from generic
  runtime warnings or a requested accelerator name. Retain historical labels
  as reported, and collect explicit hardware metadata on the next run.

## Official-source routes and decisions they change

Verified on 2026-08-31; use installed signatures and matching release notes
before copying an example. These are pointers and bounded summaries, not a
vendored manual.

1. [Cloud TPU v3](https://docs.cloud.google.com/tpu/docs/v3): two physical
   TensorCores per chip, two MXUs per TensorCore, chip HBM32GiB. Keep chip,
   physical core, JAX device, process and mesh axis distinct.
2. [JAX hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html):
   specifications are per TensorCore; v3 VMEM16MiB and SMEM16KiB, with no
   SparseCore. Other generations differ. JAX's current v3 peak/BW table does
   not numerically match Cloud's per-chip figures after multiplication by two;
   preserve the source/configuration disagreement instead of inventing one MFU denominator.
3. [TPU kernel details](https://docs.jax.dev/en/latest/pallas/tpu/details.html):
   ordinary TPU `pallas_call` iteration is sequential, except explicitly
   multicore-parallel axes. Input-window reuse matters; writes to one output
   window must remain consecutive. The last two block dimensions must be
   divisible by8 and128 respectively **or** equal the corresponding whole-array
   dimensions. Layout depends on trailing axes; CUDA warp/occupancy rules do
   not transfer. Precision and generation qualifications matter.
4. [TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html):
   software-managed VMEM/SMEM, asynchronous transfers and buffering. `pl.ANY`
   is unconstrained placement, not guaranteed HBM. Budget buffers with scratch
   and accumulator lifetimes. Default pipelining already exists; explicit
   lookahead is an experiment, not a mandatory optimization.
5. [Pallas matmul tutorial](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html):
   BM/BK/BN blocks are compiler-tiled over MXUs. The hardware array dimension
   is not a requirement to submit only that-sized Pallas dot. Accumulator
   initialization, reduction grid order, reuse and tile selection are independent concerns.
6. [Pallas distributed TPU](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html):
   remote DMA has send/receive completion requirements and semaphore lifetimes.
   Custom collectives need a correctness justification and measured advantage;
   ordinary sharding is not automatically replaced by custom communication.
7. [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html): use
   explicit device-level SPMD and collectives as a reference before custom RDMA.
   Sharding weights, sharding batches and parallelizing a Pallas grid are different decisions.
8. [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html):
   distinguish `pallas_call` and newer `pl.kernel` APIs. `pallas_call` checkify
   support was removed in0.10.1; layout-pass defaults changed in0.10.2. Do not
   import GPU-only or unreleased migration advice into a pinned TPU runtime.
9. [TPU InterpretParams](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html):
   `interpret=pltpu.InterpretParams(...)` simulates TPU memory/DMA/synchronization;
   plain `interpret=True` runs an HLO interpretation. Race detection needs to be
   enabled and is not exhaustive proof. Neither mode proves actual-device
   compilation, hardware synchronization behavior, or speed.
10. [JAX profiling](https://docs.jax.dev/en/latest/201/profiling.html) and
    [benchmarking](https://docs.jax.dev/en/latest/benchmarking.html): synchronize
    device outputs, separate compile/transfer/warm execution, and inspect the
    composed caller in traces. Capture the actual device processes.
11. [Closed-over constants](https://docs.jax.dev/en/latest/internals/constants.html):
    lowering/constant folding and sharding behavior depend on version/config.
    Record parameter delivery and test runtime arguments; do not assume a
    closure is either always wrong or identical to the production interface.
12. [JAX numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs):
    mathematically equivalent compiled graphs need not have identical floating
    outputs. Test a JAX-only decomposition before diagnosing replacement kernels.

## Next bundled experiment, after correcting the harness

1. Record original whole-model JAX, segmented JAX and same-suffix JAX controls,
   including a zero-replaced-block path. Save direct cross-comparisons.
2. Cross Dense implementation and LN implementation independently on block1:
   JAX/JAX, Pallas/JAX, JAX/Pallas, Pallas/Pallas. Change first/second sublayers
   separately when attributing an operator-specific effect.
3. Separate Dense output/bias rounding choices, LN statistic precision and
   fusion boundary. Preserve epsilon, residual/ReLU order and all masks.
4. Test stress categorical inputs **and** legal-move scrambles/real frontiers;
   save seeds, generation method, input hashes and action ordering.
5. Report finite, max/mean abs, RMSE, relative-error floor, cosine, pairwise
   equality, argmin, masked global top-K and ranking margins. Keep argmax as an
   explicitly named auxiliary diagnostic. Choose promotion gates before timing.
6. Profile passing candidates: input encoding, Dense, normalization, output head,
   composed full model and real chunk scans. Compare equal tiles for attribution,
   then each implementation's best valid tiles for engineering selection.

No new TPU run was launched. No new latency or search-quality result is claimed.

## Requirements for the maintained plugin

- Keep a compact entrypoint and load topic references only as needed: runtime and
  topology; shapes/layout/precision; Pallas kernels and DMA; validation; timing
  and scaling; project evidence and update procedure.
- Store universal method separately from this repository's measured configurations.
  Preserve BN/LN families, `MOVE_COUNT` head semantics and logical/storage shapes.
- A lesson carries source, date, runtime, hardware, input/shape scope, evidence
  state (`measured`, `documented`, `hypothesis`, `superseded`) and the test that
  would invalidate it. Preserve negative results instead of deleting history.
- Evaluate the plugin on realistic counterexamples: reversed score direction,
  missing JAX boundary control, padded LN width, capture versus runtime weights,
  compile-rejected candidates, microbench-to-full-model promotion, and obsolete APIs.
- Update after a completed experiment or verified source change. Recheck prior
  regression scenarios, version the plugin and publish a scoped change. This is
  an evidence-triggered workflow, not permission for unattended experiments or
  an automatic update schedule.
- No mandatory MCP server, credentials, cloud provisioning, or rewrite of the
  working BN implementation is needed for the first release.

Plugin structure approval and implementation remain pending; this document
preserves the research independently of that packaging decision.
