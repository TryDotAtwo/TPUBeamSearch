# Arithmetic follow-up: distinguish layout, rounding and caller overhead

2026-08-31. Read-only source/HLO/data investigation after arithmetic A/B v1.
No new TPU run, kernel fix, changed acceptance threshold or inference speedup
is established by this note. The completed measurement remains
[arithmetic v1](../../test_results/kaggle_layernorm_arithmetic_v1/report.md),
published at `163b7fbca32a4e5d9bcb0a241e546da3024fcc5b`.

## 1. The missing full candidate is narrower than “all Dense on Pallas”

`experiment_configs()` includes `cross-late-jax` in the block screen but omits
it from the full list. That list selects only `cross-bf16_before_bias-jax`
and `cross-jax-mean_jax` among cross implementations.

`full_call()`'s cross path keeps embedding, input Dense/LN and output head in
JAX; it replaces the twenty residual Dense operations. A full-forward result
for this candidate would therefore not establish an all-Pallas network.
The corresponding `cross-jax-jax` full-builder control should accompany it.

Source: [benchmark](../../benchmarks/stream1_layernorm_arithmetic.py),
`full_call` lines 190–200 and config selection lines 240–249 at the source pin.
The existing late/JAX block is close, not exact: same-suffix Q exact fractions
98.8086% legal / 98.5034% stress. Full quality remains unmeasured.

## 2. A specific HLO-informed LN candidate

The saved standalone JAX LN HLO exposes more than a BF16/FP32 switch.
Let `B` denote BF16 conversion and `F` FP32 conversion. For width 1024, the
visible schedule suggests:

```text
xf       = F(x)
mean     = B(sum(xf) * 2^-10)
centered = xf - F(mean)
variance = B(sum(centered * centered) * 2^-10)
invstd   = B(rsqrt(F(variance) + F(B(epsilon))))
output   = B((centered * F(invstd)) * F(gamma) + F(beta))
```

Evidence: [saved HLO](../../test_results/kaggle_layernorm_arithmetic_v1/arithmetic_ab/hlo/legal_scrambles-first-ln-jax_reference-False.txt),
lines 70–81 (mean), 90–100 (centered variance reduction), 103–115
(variance/epsilon/rsqrt), 118–137 (affine/output).

The existing `fp32_statistics=True` path does not round mean, variance or
invstd to BF16 and uses FP32 epsilon. Conversely, BF16 `mean_mode="jax"`
changes reduction/division placement but leaves BF16 expression-level
centering, squaring and affine operations.

This schedule is a **hypothesis**, not proven target equivalence: several HLO
FP32 operations carry `float_type_correction_info.original_type="BF16"`.
The dump is not final machine-code evidence. Returning intermediate statistics
can itself alter compilation, so instrumented comparisons cannot silently
replace the uninstrumented standalone JAX oracle.

JAX's [TPU kernel documentation](https://docs.jax.dev/en/latest/pallas/tpu/details.html#elementwise-operations)
recommends promoting low-precision operands before vector elementwise work.
That supports an explicit mixed schedule, not an assumption that FP32
everywhere matches the existing checkpoint execution.

For normal finite values away from overflow/subnormal boundaries, BF16 rounding
commutes with exact power-of-two scaling. Thus `sum_div` versus `jax` alone is
a weak discriminator at production width 1024; non-power-of-two padded width
is a separate synthetic arithmetic/layout test, not a changed production model.

## 3. Predicate failure: sites to isolate, not a verified fix

The standalone LN constructs one column predicate and uses it at three sites:
input masking, centered-value masking, output masking. See
[kernel](../../src/tpu_beam_search/stream1_layernorm_pallas.py), lines 149, 151, 160.
For logical/storage width 1024, all lanes are valid, but the iota/compare still
exists at expression level. No evidence guarantees its removal before layout
inference.

The leading suspect is column-predicate expansion
`[1024] -> [1,1024] -> [128,1024]`.
Pinned JAX lowering emits shape-cast/broadcast for this pattern and `arith.select`
for the selection:
[broadcast lowering](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/pallas/mosaic/lowering.py#L2488),
[select lowering](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/pallas/mosaic/lowering.py#L3647).
The trace identifies boolean broadcast, not which of the three mask sites.

The discriminating probes are: unchanged failure control; no redundant mask
when logical width equals storage width; restore each mask independently;
minimal BF16 versus FP32 `where`; and directly constructed 2D predicate versus
rank expansion. FP32 selection of BF16 inputs followed by BF16 conversion
does not change the selected finite BF16 value or zero, making it a useful
layout-only comparison. Padded-width masks must remain intact.

### A tempting compiler switch is not an effective TensorCore A/B

In JAX 0.10.2, the custom-call route explicitly forces layout passes for
TensorCore kernels:

```text
needs_layout_passes = needs_layout_passes or not device_type
```

TensorCore uses `device_type=None`; see
[pinned source](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/tpu_custom_call.py#L595).
Setting `CompilerParams(needs_layout_passes=False)` therefore does not disable
this route's TensorCore layout passes. Count only changes in effective emitted
configuration as distinct experimental arms.

`shape_invariant_numerics=False` is transported to the custom-call config, but
the inspected public source does not establish its detailed rounding behavior
or a connection to this predicate error. It is not an evidenced fix.

## 4. The microbenchmarks are noisy; the full graph is already one call

Recomputed from unchanged raw samples; spread is `(max-min)/median`, not a
confidence interval:

| Legal case | Batch | Median ms | Min–max ms | Spread |
|---|---:|---:|---:|---:|
| JAX first Dense | 4096 | .29192 | .24840–.94399 | 238.3% |
| JAX second Dense | 4096 | .26108 | .23433–.31132 | 29.5% |
| JAX first LN | 4096 | .24359 | .23374–.27313 | 16.2% |
| JAX second LN | 4096 | .28333 | .25505–.36970 | 40.5% |
| JAX block | 4096 | .39290 | .37827–.47929 | 25.7% |
| Late Dense/JAX LN block | 4096 | .56534 | .54822–.62219 | 13.1% |
| Original full JAX | 16384 | 11.50652 | 11.47490–11.52268 | .42% |
| Typed full JAX | 16384 | 11.51167 | 11.46069–11.70353 | 2.11% |
| Early Dense/JAX LN full | 16384 | 16.32125 | 16.18954–16.44499 | 1.57% |
| Early/FP32 per-block full | 16384 | 29.11755 | 29.06776–29.24309 | .60% |

Several microprobe sequences fall across seven samples despite three warmups.
This demonstrates timing instability, not its cause. The full-model gaps are
larger than the observed repeat spreads.

`measure()` times one compiled call plus `block_until_ready()`. Inputs are
resident and compilation is separate, but wall time still includes host
dispatch/synchronization. **Full inference is already one compiled invocation**;
twenty residual Dense operators are not twenty Python launches.
Neither pure host overhead nor device memory traffic is isolated yet.

The four legal standalone Dense/LN medians sum to 1.07992 ms, whereas the
composed JAX block is .39290 ms. Summing those microbenchmarks would confound
fusion, invocation overhead and execution scope.

The profiler is guarded by `if exact and not interpret` in the current runner,
so no failed candidate or baseline received a diagnostic trace. Separate
diagnostic profiling from optimization acceptance: trace baseline and a
small predeclared set of compiled candidates even if Q fails; keep failed
quality flags and null eligible speedups unchanged. Profile outside timing.

Before changing compilation to a large scan, a small queue of calls to the
**same compiled executable**, retaining and synchronizing every output, can
test dispatch amortization. Label this queued throughput, not single-call
latency or a real 128-chunk beam workload. A compiled loop can change fusion
or remove invariant work, so it introduces an additional control problem.
[Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html),
[profiling](https://docs.jax.dev/en/latest/profiling.html).

## 5. Duplicates have a useful mathematical lower bound

The cyclic depth generator produces 1821 zero-walk and 1821 one-move rows in
the first 16384 legal rows. These 3642 rows contain at most 31 distinct states:
identity plus thirty moves. Consequently at least **3611 rows (22.04% of the
whole corpus)** duplicate a representative from these two strata alone.
This is a lower bound derived from the generator, not a measured unique count.
Longer walks may add duplicates.

Equal large quotas of unique states at depths 0/1 are impossible. Preserve
the existing corpus as a reproducibility anchor; use shallow exhaustive cases
separately and report achieved counts for deeper deduplicated strata. State
uniqueness and `(state,last_move)` uniqueness differ for inverse-mask analysis.
Add strict-below-K core preservation and tie-group diagnostics alongside the
existing stable flat-ID comparisons, without changing the exact-Q gate.

## 6. Hardware and headroom are not the old v3 model

Pinned JAX maps `TPU v5 lite` to `TPU_V5E` explicitly:
[device-kind mapping](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/pallas/mosaic/tpu_info.py#L113).
Cloud's [v5e architecture](https://docs.cloud.google.com/tpu/docs/v5e)
documents one TensorCore and four MXUs per chip, with a 197 TFLOP/s BF16 peak.
Do not continue tuning against the earlier two-MXU v3 mental model.

The model's Dense-equivalent work at batch 16384 is
`2*16384*(3600*1024 + 20*1024*1024 + 1024*30) = 808,997,355,520` operations.
Dividing by legal-corpus typed-JAX full latency of 11.51167 ms gives
70.28 TFLOP/s, about 35.67% of that
published peak. This is a shape-derived convention over whole-model time,
**not measured MXU utilization or evidence of an attainable 2.8x speedup**;
LN, embedding, on-device data movement, padding and scheduling also consume time.

The current [JAX hardware table](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
lists 128 MiB VMEM for v5e, while earlier experiments hit a 16 MiB *scoped*
compiler allocation limit. Those are different quantities. This observation
does not authorize increasing budgets blindly, and does not explain the
current predicate error. Record runtime hardware info and effective allocation
flags before revisiting VMEM-limited fusion.

## 7. Expert consultation and decisions

Consulted `multigpu_beam` and `multigpu_mlp` together on 2026-08-31 with model,
runtime, corpus, rounding, timing and control results, and the pinned v1 report.
Their advice is project-peer review, not an independent TPU experiment.

Both prioritized full late-Dense/JAX-LN, boundary-matched controls, then
HLO-informed LN and richer legal/tie diagnostics. These recommendations match
the local source/data audit. Original monolithic JAX remains the acceptance
oracle; segmented and same-suffix paths serve causal diagnostics only.
Do not subtract aggregate errors as if they were an additive decomposition.

The advice to profile **only accepted candidates** is not adopted for failure
diagnosis: that is the current harness behavior and left us without traces.
Diagnostic profiles of rejected candidates remain explicitly ineligible for
optimization claims. The suggestion to use a compiled scan is deferred behind
same-executable queuing because a scan changes compilation boundaries.

Proposed bounded continuation: one bundle containing the minimal predicate
probes, standalone mixed-LN comparison, full late/JAX and matching controls,
and explicitly diagnostic profiles. Keep the existing corpus/acceptance anchor,
BN path and one-TPU-session policy. Implementation/launch awaits confirmation;
no experiment source or automation was changed in this research step.
