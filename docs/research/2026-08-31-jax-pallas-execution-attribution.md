# Why the current JAX inference beats our Pallas kernels

2026-08-31. Source/profile audit of the completed arithmetic follow-up, not a
new TPU run. Evidence snapshot: repository `fb248f5`, benchmark source
`d58cf9fd8e86ec145c6bbc4f6c7f5aff489d6e21`. Actual target: **one TPU v5 lite**,
JAX/jaxlib0.10.2, libtpu0.0.42.1. State150, categories150, embedding24,
hidden1024, ten residual blocks, Q30. BN and production defaults are unchanged.

## Answer

The comparison is not two interchangeable matmul implementations. XLA's JAX
path combines neighboring vector work with Dense and chooses a different
layout/window schedule. Our current Pallas Dense is an opaque, smaller tiled
unit with explicit BF16 output and separate downstream normalization. The
device profiles locate the performance loss; they do not yet divide it into
fusion, tile size, layout, and final instruction-scheduling costs.

Separately, certain redundant LayerNorm predicates cause a large measured
device penalty. Neither Python dispatch nor a missing default double buffer
explains these observations. No evidence establishes an intrinsic Pallas
performance ceiling, or that a larger fused kernel alone will solve the gap.

## 1. Twenty Dense-containing operations: 3.85ms versus 8.34ms

From the legal16384 full-model device traces, summing each operation's mean
over three executions:

| Implementation | Sum of 20 selected operations, ms | Mean per operation, us |
|---|---:|---:|
| Original runtime JAX |3.854593334|192.729667|
| Typed runtime BF16 JAX |3.849192343|192.459617|
| JAX full-builder control |3.849276433|192.463822|
| Late Pallas Dense + JAX LN |8.337010651|416.850533|

For the three JAX rows, select `convert_reduce_fusion.1, .3, ..., .39` in
[profile_summary.json][profiles]. Their compiled computations contain the
twenty residual dots. These names are **not portable selectors**: in the
Pallas-Dense model the same names denote separate downstream reductions.
For the Pallas row, select category `pallas_dense`, verifying twenty operations
per forward. Embedding, input Dense and output head are excluded. The first
residual JAX operation includes the fused input-layer LN affine/ReLU producer.

The JAX timings include dot, bias and the following LN sum, often also the
preceding LN affine/ReLU or residual producer. They are **not pure GEMM
latencies**. Pallas spends 2.16x as long on its Dense operations while doing
less fused work; its separate following sum reductions add 0.148724739ms.
The full module medians are 10.970597422ms original and 15.533120000ms late/JAX.
The 4.4824ms selected-operation difference strongly localizes the 4.5625ms
module difference, but is not an exact causal accounting: other fusions and
layouts change, and sums of means are not differences of medians.

Using only `20 * 2 * 16384 * 1024 * 1024` useful Dense FLOPs gives roughly
178.3 TF/s for the JAX-containing operations and 82.4 TF/s for Pallas Dense.
These are workload/time conventions, **not measured MXU utilization** or
full-model FLOP rates; the extra fused vector work is not in the numerator.

## 2. The compiler boundary changes both work and a numerical pathway

Inspect [typed full compiled HLO][typed-hlo], lines257-276:

- A preceding `add_maximum_fusion` is nested in the Dense-containing fusion.
- The dot is convolution-lowered; bias addition is represented in FP32.
- The sum used for the following LN mean consumes that FP32 biased result.
- A separate BF16 conversion of the same result is returned alongside the sum.

By contrast, [late/JAX compiled HLO][late-hlo], lines2067-2081, has a separate
producer, Pallas custom call with `allow_input_fusion: []`, BF16 Dense output,
and then the JAX sum reduction. Its activation layout is
`{1,0:T(8,128)(2,1)}`, versus `{0,1:T(8,128)(2,1)}` in the JAX fusion.
Weight slices are staged in memory space `S(1)` before the custom call.

The concrete **outer-HLO hypothesis** is that the monolithic mean can use the
biased FP32 value while the hybrid mean sees materialized BF16. However,
`float_type_correction_info.original_type=BF16` annotates the FP32 addition.
This audit does not establish the final TPU rounding/type-correction behavior,
nor prove that this branch explains all Q drift. Saved outer HLO and embedded
pre-layout StableMosaic are not final machine instructions.

An explicit JAX boundary is therefore a useful control, not a proposed
production optimization. JAX documents that an
[optimization barrier prevents fusion across it][barrier]. Compare ordinary
JAX, JAX with input/output Dense barriers, and Pallas on identical resident
inputs. Verify the actual compiled boundaries. Also retain separately compiled
JAX Dense/LN controls: a barrier does not guarantee every other layout and
schedule matches Pallas. JAX itself documents that
[JIT transformations can change exact numerics][jit-numerics].

For attribution, record direct pairwise mismatch witnesses at Dense output,
mean, variance, inverse standard deviation and affine output. Returning these
intermediates can itself change fusion: check the instrumented JAX final Q
against the untouched monolithic oracle before drawing conclusions.

## 3. What the current Pallas Dense schedule does and does not prove

[The implementation][dense-source] uses BM128/BK256/BN512 in this run:
grid `(128, 2, 4)` for `[16384,1024] @ [1024,1024]`. Each K step forms an FP32
partial dot and adds it to an FP32 scratch accumulator; bias is added at the
last step before conversion to BF16. All these residual dimensions divide
exactly: there is no residual Dense padding overhead to blame here.

Decoded embedded StableMosaic preserves a zero-initialized partial matmul
followed by a scratch add. Whether final lowering uses the scratch directly as
an MXU accumulator remains unverified. The Pallas custom-call scoped VMEM
allocation is 1,179,648 bytes, versus about 8.38-12.03MB for the JAX
Dense-containing fusions. These are compiler-scoped allocations, not physical
VMEM capacities or direct evidence of a capacity bottleneck.

Default Pallas TPU pipelining already uses two buffers for inputs and outputs;
[the documentation describes this default][pipeline]. It is incorrect to
attribute this result to an absence of double buffering. Larger tiles could
improve reuse, but also increase live state; the official
[matmul tutorial explains that tradeoff][matmul].

One sanity check rejects an overly literal bandwidth explanation: charging
every tile-window reread to HBM gives 352.25MiB per Dense (64MiB X, 256MiB W,
32MiB Y and 0.25MiB bias). At the tutorial's 819GB/s reference this predicts
0.45099ms, already above the observed 0.41685ms. Thus that accounting cannot
be treated as measured HBM traffic. Staged weights and compiler-chosen memory
placement matter. Trace `bytes_accessed=0` for Pallas means missing metadata,
not zero bytes, and JAX's static traffic estimate is not a hardware DMA counter.

A discriminating tile sweep changes BM alone first (128/256/512 with BK256,
BN512), then BN separately. Changing BK changes the partial-sum arithmetic as
well as performance, so a BK1024 arm must be labeled a joint arithmetic and
scheduling experiment, not pure layout tuning. No such new sweep has run yet.

## 4. LayerNorm has a separate predicate-related performance problem

Holding JAX Dense and the full legal16384 model fixed:

| Pallas LN | Sum of 20 LN operation means, ms | Module median, ms |
|---|---:|---:|
| Legacy BF16, redundant masks removed |1.983675443|12.084825000|
| Legacy BF16, promoted FP32 `where` |16.350091511|26.448022578|
| Mixed arithmetic, masked |10.812527969|20.915475156|

The first mask change adds about14.366ms in LN and14.363ms to the module;
mixed masked adds8.829ms in LN and8.831ms to the module. The penalty is
localized on the device. Its exact low-level cause (relayout, register
pressure, extra movement or instruction schedule) is not established.

Checkpoint operator arms narrow the mixed-LN slow case to the centered-value
mask: `center/all` queued times are about0.160ms, while `none/direct2d` are
about0.083/0.077ms. Full mixed-direct2D is still unmeasured. Removing masks is
valid only when logical width equals storage width; the width130 population
and output masks remain mandatory.

These faster arms are **not exact LN implementations**. Equal aggregate error
metrics between arms do not prove pairwise tensor equality. Matching HLO
dtype boundaries does not identify the remaining reduction/rounding mechanism.
The reference uses centered variance, not `E[x*x] - E[x]*E[x]`.

## 5. Correcting the historical embedding comparison

The old label "Pallas embedding gather" did not describe a custom gather
kernel. [The source][input-source] calls ordinary JAX embedding indexing and
reshape, followed by Pallas Dense/LN. The old input A/B compared whole input
prefixes on repeated synthetic states; it did not isolate gather performance.
Its raw timings remain historical measurements, not evidence against a
custom flat-output gather. This corrects the interpretation without relabeling
or modifying the old JSON.

In the original current full trace, `gather_fusion` takes4.258747917ms and
embedding-flattening `reshape.65` takes1.217208307ms. Their5.475956224ms sum is
about half the10.9706ms module. The other index `reshape.68` is only0.023565052ms
and is not the embedding flatten. Since this work is shared, it does **not**
explain the JAX/Pallas candidate gap. It is an independent optimization target.

The exact flat-output contract is:

```text
F[b,t] = BF16(E[int32(states[b,t//24]), t%24])
states: uint8[B,150], values 0..149
E: runtime FP32[150,24]; F: BF16[B,3600]
```

Preserve position-major ordering, classes128-149, runtime parameters and
Dense logical K3600. A separately labeled runtime-BF16 parameter control may
be retained, but must not silently replace the original FP32 delivery contract.
Existing compiled JAX already casts E to BF16 before
gather, so simply precasting E is not a new optimization. A scalar flat gather
may trade the reshape for large generated indices; test actual HLO/live memory.
A tiled flat-output Pallas gather must first compile and exactly match F,
then be evaluated inside the unchanged full JAX suffix. No removable-time or
full-Q speedup is inferred from isolated gather savings. Embedding folding
into Dense changes rounding and contraction order and is a different experiment.

## Decision and proposed next bundle

Keep original JAX as the accepted full model. None of these findings relaxes
the finite, exact monolithic Q gate on both16K corpora, actual32K confirmation,
or the distinction between global top-K proxies and distributed beam search.

The bounded proposal has three parts: matched Dense/boundary controls and
independent tile axes; direct2D/unmasked LN with numerical witnesses; exact
flat embedding alone and inside full JAX. Use the existing hashes, matched
interleaved timing, retained queued outputs, diagnostic profiles even for
rejected cases, pinned GitHub source and at most one TPU session. No new
implementation or TPU launch is claimed by this document.

Three independent source/profile audits and the MLP/beam project experts
reviewed the reasoning. Their advice supports controls and cautions; it is not
target evidence. The TPU coding skill's separate arithmetic/execution contracts
prevented treating a faster rejected operator or different HLO layout as an
accepted numerical or performance fix.

## Evidence and reproducibility

The [completed report][report] retains runtime/hash provenance, raw results,
all profile paths, compile failures and unchanged acceptance decisions.
Revalidate the saved profile extraction from the repository root:

```text
python test_results/kaggle_layernorm_followup_v1/profile_summary.py --check
python test_results/kaggle_layernorm_followup_v1/profile_summary.py --self-test
```

For the Dense table, select the exact named operations only in the three JAX
rows and category `pallas_dense` only in the late/JAX row; sum
`operations_by_name[*].ms_per_forward`. Use category `pallas_ln` for the LN
table. This avoids double-counting nested host events or mistaking the same
operation name for the same computation across compiled programs.

Fresh checks for this audit passed: ten trace summaries / thirty TPU module
calls, three profile-parser tests, numerical summary regeneration and its
negative tests, the259-file artifact manifest, all six local evidence links,
and all four Dense table sums with20 unique operations / three samples each.
This is analysis/artifact verification; no new kernel or TPU result is claimed.

[profiles]: ../../test_results/kaggle_layernorm_followup_v1/profile_summary.json
[report]: ../../test_results/kaggle_layernorm_followup_v1/report.md
[typed-hlo]: ../../test_results/kaggle_layernorm_followup_v1/arithmetic_followup/hlo/full-16384-legal_scrambles-typed_runtime.compiled.txt
[late-hlo]: ../../test_results/kaggle_layernorm_followup_v1/arithmetic_followup/hlo/full-16384-legal_scrambles-late-dense-jax-ln.compiled.txt
[dense-source]: ../../src/tpu_beam_search/stream1_pallas.py
[input-source]: ../../src/tpu_beam_search/stream1_layernorm_pallas.py
[barrier]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
[jit-numerics]: https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs
[pipeline]: https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html
[matmul]: https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
