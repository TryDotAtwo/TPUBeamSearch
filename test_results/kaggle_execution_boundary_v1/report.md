# Execution-boundary v1: exact Pallas embedding wins; Dense/LN remain rejected

Completed2026-08-31. [Kaggle notebook](https://www.kaggle.com/code/trydotatwo/tpu-execution-boundary-ab),
[frozen protocol](../../docs/research/2026-08-31-execution-boundary-bundle.md),
[launch evidence](launch.md), [raw result](execution_boundary/stream1_execution_boundary.json).

## Result and scope

**Replacing only embedding lookup with `pallas_banked`, leaving the arithmetic
network on JAX, accelerates the full checkpoint by1.267–1.278x at16K and
1.296–1.300x at32K.** All Q values are finite and elementwise exact against the
original monolithic `jax_model.apply` on both corpora at both batch sizes.
Max/mean absolute error and RMSE are0; cosine, exact fraction, minimizing
argmin agreement, top-K identity overlap and order agreement are1, including
the inverse-mask diagnostic. This is measured sample exactness, not a proof
over every possible input or a signed-zero/NaN bitwise equivalence theorem.

The simpler `jax_tiled` lookup is also exact and faster, but loses to banked
Pallas in every paired round on both16K and32K corpora. `jax_flat` is exact but
about49x slower than original at16K; it is not a performance winner. All
residual Dense and LayerNorm replacement arms fail the unchanged exact-Q gate.
No gate was loosened, rejected candidates have no eligible speedup, and no
inference default or BN path was changed.

This is **one active TPU device**, not8-device scaling or full beam search.
Next useful measured target is this exact hybrid in the real caller; no new
Kaggle job is submitted by this terminal analysis.

## Runtime, provenance and complete matrix

Runtime is Python3.12.13, JAX/jaxlib0.10.2, libtpu0.0.42.1. Eight devices report
`TPU v5 lite`; only device0 is active. This confirms v5e-generation execution,
not the requested `v3-8` scheduling label. No hardware utilization or TFLOPS
claim is made.

Source `45062324d368f4849adb6d572d21d54f75854d79`; launcher
`51c8b3a512e650df83939d322c78bd715cfd8221`. The recorded checkpoint, original
model source, puzzle and both32768-input hashes match
[arithmetic follow-up v1](../kaggle_layernorm_followup_v1/arithmetic_followup/stream1_layernorm_followup.json).
The analysis verifies recorded hashes against that prior JSON; it does not
download and independently rehash the external checkpoint again.

| Asset | SHA256 |
|---|---|
| Checkpoint | `2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2` |
| Original model source | `6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e` |
| Puzzle | `01c616cb943d574d1b63109f350b30c7710656e53e2a5eaebb5d50ed0e495ff0` |
| Legal32768 | `b7ff6754f3cfbd0c531d14e00b7f516adba8f412c3ee9d93ede5464cc0730fc1` |
| Stress32768 | `610be0697c4a087f2a66dbcd5622bfdcc13e0fac9efc0106a3d62fa28989ea5d` |

Architecture:150 state positions,150 categories, embedding24, hidden1024,
ten residual blocks,30 minimizing-Q outputs. Embedding150x24 does not mean
24 categories. One state gives30 scores in one forward.

All106 result rows execute successfully:56 timed operators,4 observed-node
cases,4 direct controls,30 full16K candidates,8 baseline rows across16K/32K,
and4 actual32K promotions. All12 interleaved timing groups succeed.
**Zero compilation, VMEM, runtime or recorded case errors.** One diagnostic
profile has an internal timestamp inconsistency, documented separately below;
that is not a benchmark crash or reason to rerun this kernel.

Original baseline receives runtimeFP32 parameters; typed JAX receivesBF16.
Candidates receive runtimeFP32 embedding andBF16 remaining parameters. Lookup
casts/preparation stay inside the compiled/timed call. There are no captured
full-model constants and no untimed precomputed embedding tables.

## Full-model timing and independent32K confirmation

Each comparison compiles all members first, uses resident arguments,5warmups,
then12 forward/reverse interleaved synchronous samples. Numbers below are
medians in milliseconds; brackets give observed min–max, not a confidence
interval. Throughput counts input states, not the30 output scores.

| Corpus / batch | OriginalFP32-runtime JAX | TypedBF16-runtime JAX | JAX tiled embedding | Pallas banked embedding | Pallas / original throughput |
|---|---:|---:|---:|---:|---:|
| Legal16K |11.5929 [11.3744–11.9168]|11.4122 [11.3318–11.6803]|10.2066 [9.8293–10.3873]|9.1513 [8.9699–9.5632]|1.2668x|
| Stress16K |11.6772 [11.4992–11.8077]|11.5670 [11.3626–11.9817]|10.0986 [9.8189–10.4488]|9.1341 [9.0101–9.3653]|1.2784x|
| Legal32K |24.1225 [23.9998–24.2732]|24.1130 [23.9829–24.2785]|20.9149 [20.8384–21.2024]|18.5493 [18.4365–18.7866]|1.3005x|
| Stress32K |24.1381 [23.9596–24.4298]|24.1819 [24.0046–24.3471]|20.9898 [20.9099–21.1007]|18.6259 [18.5304–18.7301]|1.2959x|

Banked Pallas throughput is1.790/1.794million states/s at16K and1.767/1.759million
at32K (legal/stress). Against typed JAX it wins1.247/1.266x at16K and
1.300/1.298x at32K. Thus this is not a misleading comparison against only
runtimeFP32 weights. Each comparison with either baseline wins12/12 rounds.
The paired original/Pallas ratios span1.189–1.295 and1.231–1.305 at16K;
1.283–1.315 and1.285–1.311 at32K. Raw samples and per-round ratios are in
[summary.json](summary.json); no statistical population claim is inferred
from these12 rounds.

`jax_tiled` gives1.136/1.156x at16K and1.153/1.150x at32K against original.
At16K, `jax_flat` takes570.099/570.119ms, with eligible throughput ratios
0.02033/0.02048. Exactness permits comparison, not a claim of acceleration.
Only the top two non-controls were promoted, and **both genuinely executed32K
on both corpora**;32K was not extrapolated from16K.

Eight retained calls to the same executable amortize full legal16K cost to
11.051ms original,11.008ms typed,9.491ms tiled and8.644ms banked. At legal32K
the corresponding values are23.627,23.608,20.450,18.052ms. Queueing reduces
call overhead but does not remove the measured device difference. These are
five queue batches of eight identical-input calls, **not real128-chunk scan**.

## Why embedding helps: device and memory evidence

Three-call legal16K device profiles, separate from the timing rounds:

| Implementation | Device module mean, ms | Relevant device operations |
|---|---:|---|
| Original JAX |10.9668|gather4.2585 + embedding flatten reshape1.2160ms|
| Typed JAX |10.9423|gather4.2585 + embedding flatten reshape1.2096ms|
| JAX tiled |9.4181|128 small gathers total3.0864ms; tiled reshape/update/loop work remains|
| Pallas banked |8.5641|one banked kernel2.4136ms; runtime table-preparation loops about0.6659ms inclusive|

The banked implementation produces aligned `[16384,3712]` storage directly;
logical embedding width stays3600. The input Dense consumes that output in
the compiled graph. Original rank-three gather/flatten operations are absent
from its device trace. Preparation of the two phase-indexed lookup banks
remains runtime work and is not hidden from measurement. The rest of the
Dense/LN network remains JAX; twenty residual Dense-containing JAX operations
sum to about3.8503ms in the banked case versus3.8492ms in typed JAX.
The measured module difference is therefore concentrated in the lookup/layout
path, not a suddenly faster residual network.

Evidence: [banked compiled HLO](execution_boundary/hlo/full-16384-legal_scrambles-embedding-pallas_banked.compiled.txt)
contains runtime table-preparation `while.3`/`while.2`,
`stream1_flat_embedding_banked.1`, and input `convert_reduce_fusion.41` consuming
its result. [Tiled HLO](execution_boundary/hlo/full-16384-legal_scrambles-embedding-jax_tiled.compiled.txt)
contains a128-iteration lookup loop. [Flat JAX HLO](execution_boundary/hlo/full-16384-legal_scrambles-embedding-jax_flat.compiled.txt)
contains broad `[16384,3600]` scalar-index operations and58,982,400-element
index/output reshapes; flattening source code alone does not guarantee a cheap
lowering. We do not assign a precise memory-bandwidth cause without counters.

Compiler static temporary allocation drops from723.660MiB original /
723.629MiB typed to123.967MiB banked at16K. At32K it drops from1440.367 /
1440.336MiB to240.429MiB. These are static allocation reports, **not measured
HBM traffic, bandwidth or peak live hardware memory counters**.

Isolated lookup at4K would have selected differently: reference1.717/1.718ms,
typed1.788/1.708ms, tiled1.344/1.352ms, banked1.699/1.666ms, flat117.491/117.466ms.
All ten lookup outputs are exact. The banked win appears in the larger full
graph; its runtime preprocessing cost and the consumer layout matter. Do not
promote an isolated4K ranking to an end-to-end choice.

## Dense: tiling improves time; boundaries still break full exactness

Matched4K Dense results (legal/stress):

| Dense arm | Synchronous ms | Queued ms | Direct mismatch count out of4,194,304 |
|---|---:|---:|---:|
| JAX none |0.2720 /0.2440|0.0806 /0.0781|0 /0|
| JAX pre |0.2755 /0.2495|0.0783 /0.0712|0 /0|
| JAX post |0.2796 /0.2451|0.0776 /0.0725|0 /0|
| JAX both |0.2884 /0.2542|0.0805 /0.0699|0 /0|
| Pallas128/256/512 |0.4216 /0.3880|0.2029 /0.1996|20 /31|
| Pallas256/256/512 |0.3525 /0.3226|0.1400 /0.1356|20 /31|
| Pallas512/256/512 |0.3220 /0.2863|0.1078 /0.1034|20 /31|
| Pallas128/256/1024 |0.3637 /0.3360|0.1529 /0.1490|20 /31|
| Pallas128/1024/512 |0.3557 /0.3334|0.1441 /0.1414|0 /0|

Tile notation isBM/BK/BN. ChangingBM/BN atBK256 preserves the same reported
error metrics; this alone does not prove pairwise array identity. BK1024
removes the observed standalone Dense discrepancies on these two inputs and
is exact after feeding **the same separately compiled LN**. It also changes
the reduction schedule, not just memory tiling. It remains slower than JAX
standalone and fails the full-model gate.

The decisive control is all-JAX: separate Dense then separate LN differs from
composed JAX Dense+LN in26,624 /25,600 elements (exact99.3652% /99.3896%). Max
error is0.121094 /0.093750; RMSE0.002964 /0.002027. Adding a post-Dense JAX
barrier gives these same aggregate errors. BK1024 Pallas+JAXLN does too,
despite its exact standalone Dense and exact same-compiled-LN witness.
This directly demonstrates an execution-boundary effect without any Pallas LN.
It does not justify subtracting aggregate errors or claiming arrays equal
where a pairwise comparison was not recorded.

[Composed JAX HLO](execution_boundary/hlo/dense_ln-legal_scrambles-jax-none.compiled.txt)
branches the FP32 biased result into the LN sum and a BF16 output conversion.
[Post-barrier HLO](execution_boundary/hlo/dense_ln-legal_scrambles-jax-post.compiled.txt)
returns a BF16 Dense result before a separate sum operation, as does
[late Pallas HLO](execution_boundary/hlo/dense_ln-legal_scrambles-late-m128-k1024-n512.compiled.txt).
Emitter/layout selection also changes. This supports the boundary explanation;
`float_type_correction_info` means outer HLO alone is still not proof of every
final machine rounding point. Exact standalone GEMM is necessary here, not
sufficient for monolithic equivalence.

Full16K residual replacements, all diagnostic and **rejected** unless marked
control exact. Exact fractions and minimizing top-K overlaps below are
legal/stress; all numbers are finite.

| Arm | Full ms legal/stress | Q exact fraction | Argmin agreement | Top-K overlap |
|---|---:|---:|---:|---:|
| JAX none, exact control |11.382 /11.572|1 /1|1 /1|1 /1|
| JAX pre |11.564 /11.693|.335832 /.179543|.927917 /.871155|.754578 /.931335|
| JAX post |11.363 /11.472|.909283 /.864549|.985901 /.980408|.944580 /.984253|
| JAX both |11.303 /11.511|.328947 /.176371|.926697 /.867920|.750610 /.929932|
| Pallas128/256/512 |16.022 /16.167|.328691 /.175916|.926514 /.867432|.750610 /.929871|
| Pallas256/256/512 |13.935 /14.062|.328691 /.175916|.926514 /.867432|.750610 /.929871|
| Pallas512/256/512 |12.929 /13.085|.328691 /.175916|.926514 /.867432|.750610 /.929871|
| Pallas128/256/1024 |13.142 /13.283|.328691 /.175916|.926514 /.867432|.750610 /.929871|
| Pallas128/1024/512 |12.990 /13.014|.328945 /.176359|.926697 /.867920|.750610 /.929932|
| JAXDense +legacyLN |12.633 /12.763|.242133 /.118477|.899048 /.830383|.654968 /.899475|
| JAXDense +mixedLN |12.532 /12.587|.258252 /.128335|.904968 /.831909|.714600 /.904114|
| JAXDense +mixedLN direct2D |12.532 /12.694|.258252 /.128335|.904968 /.831909|.714600 /.904114|

The other three of15 full arms are the exact embedding arms already reported.
Full max/mean abs, RMSE, cosine, exact fraction, top-K order, masks, ties,
compile time and every timing sample are retained losslessly in
[full_summary.csv](full_summary.csv), not collapsed into one quality metric.
For example lateBK256 has legal max2.75, mean0.103467, RMSE0.192031 and
cosine0.999974, yet top-K overlap only75.06% and exact order only0.464%.
High cosine is not a ranking-validity gate. Masked legal overlap falls to73.23%.

Device profiles quantify real tiling progress: twenty Pallas Dense calls take
8.3370ms atBM128/BK256/BN512;6.2431ms atBM256;5.2164ms atBM512;
5.4265ms withBN1024;5.1539ms withBK1024. Typed JAX's twenty Dense-containing
operations take3.8492ms and include fused vector work and LN sums. They are
not pure matched GEMMs. Whole device modules decrease15.530→12.411ms from
BM128→512, but still exceed typed10.942ms. The remaining gap is on device,
not simply Python startup or a promise that later queued calls hide it.

## LayerNorm and observed JAX nodes

All five4K LN arms compile, including direct2D and FP32-where. Legacy unmasked
exact fractions are47.981% /52.748%; mixed unmasked90.076% /87.980%. Mixed
direct2D and mixed FP32-where have the same recorded metrics. The direct
mixed-unmasked versus direct2D comparison is actually exact on both corpora.
No LN candidate matches JAX exactly.

Synchronous legal times: JAX0.2497ms, legacy0.2404, mixed0.2360, direct2D0.2346,
FP32-where0.3619. Stress:0.2336,0.2302,0.2218,0.2200,0.3441ms. Queued mixed
unmasked is0.0777/0.0807ms versus0.1605/0.1608ms for FP32-where. A mask used
as a compiler workaround has measurable cost even when numerical metrics
match; removing an aligned-width redundant mask does not solve arithmetic.

Full mixed direct2D versus unmasked has the same reported errors and almost
identical device LN totals1.8163 versus1.8170ms for20calls, module11.9191 versus
11.9200ms. They share the traced module fingerprint. This is useful measured
equivalence for this layout, not a portable promise about arbitrary widths.
Legacy LN totals1.9850ms. None is accepted for deployment by this experiment.

Four instrumented JAX observations return finite BF16 mean, variance, inverse
standard deviation and outputs; the Dense+LN observations also return BF16
Dense. Every instrumented output equals its uninstrumented comparator on the
sample. Therefore no output observer effect is detected here. Recorded node
samples are in [observation_summary.csv](observation_summary.csv). This does
not expose Pallas internal statistics or prove the original uninstrumented
machine operation ordering. The remaining LN arithmetic mechanism is unresolved.

## Ranking, compile costs and profile caveats

Legal16K contains11401 unique states, not16384 distinct beam-frontier states.
Stress16K has16384 unique states. Inputs were not deduplicated. At legal16K
unmasked K-boundary has5337 equal-score candidates; inverse masking has8039.
At32K those counts are10278/15885. Stress K-boundary ties number1337 at16K
and2543 at32K. Stable ordering is score then flat candidate ID; stress inverse
masking is a no-op because last_move=-1. No invalid candidate was selected by
the accepted variants. Global flattened top-K remains a diagnostic proxy,
not distributed-beam selection or a solved-puzzle replay test.

Compilation is excluded from steady-state timing. First legal16K compilation
takes6.168s original,6.112s typed,1.906s tiled,1.843s banked. At32K the first
compilations take13.543,13.309,2.744,2.834s respectively. Later baseline stress
compile calls are cache hits; their tiny durations are not independent cold
compile measurements. First-execution timings and all static byte counts
remain in the raw JSON/CSV.

All17 Chrome traces and17 XPlane protobufs are published. Device attribution
uses only TPU0 XLA Ops, with separate enclosing module durations. The two
tiled implementations expose inclusive `while` spans on the same lane as
their children: the analyzer verifies containment and excludes loop wrappers
from sums, retaining leaf work and module gaps. It never adds parent and
child durations or host spans together. This adaptation is regression-tested.

**16/17 traces pass the strict attribution checks.** In `jax_flat`, the second
module's Chrome duration is569392.135750us while device_duration_ps implies
569386.178750us, a5.957us discrepancy. That trace is retained but excluded from
normalized operator attribution; timestamps are not silently rescaled and
tolerances are not widened. Its unprofiled paired timing and exact-Q results
remain valid. The other profiles pass the original1ns unit crosscheck and
non-overlapping leaf checks. XPlane files are hashed but not decoded. No
precise cause is assigned to the flat trace's clock inconsistency.

## Artifacts, verification and next work

All240 API-listed outputs plus the complete Kaggle log are present:
204 HLO text files,17 Chrome traces,17 XPlane files, one result JSON, one runner
log, and the separate Kaggle log. Total48,956,285 bytes. Transient proxy/TLS/HTTP
errors during download were retried without modifying networking or restarting
the completed kernel. Each successful HTTP response was checked, existing
bytes compared, and the final manifest verified. API-signed download URLs and
local CLI network-error output are not published.

- [Download manifest](download_manifest.json): complete path list, sizes and
  SHA256 of raw bytes. These hashes are integrity checks, not remote signatures.
- [Mechanical summary](summary.json), [all operator rows](operator_summary.csv),
  [full rows](full_summary.csv), [controls](control_summary.csv),
  [observations](observation_summary.csv):106 rows,8,659 resolved CSV cells
  independently checked against raw JSON paths with no float rounding.
- [Device profile summary](profile_summary.json) and [CSV](profile_summary.csv):
 16 verified profiles, one explicit rejection, per-operation timings and hashes.
- [Runner log](execution_boundary/benchmark.log) and
  [full Kaggle log](tpu-execution-boundary-ab.log).
- [Reproducer](analyze.py) and [regression tests](../../tests/test_execution_boundary_analysis.py).

Regenerate from the repository root with Python3.10+ (stdlib only):

```text
python test_results/kaggle_execution_boundary_v1/analyze.py
python test_results/kaggle_execution_boundary_v1/analyze.py --check
python -m pytest -q
```

The analyzer reuses the prior run's lossless CSV serializer and strict
device-lane parser, verifies runtime/provenance, paired samples and frozen
promotion gates, checks raw artifact hashes and produces deterministic outputs.
New regression tests were observed failing before implementation for gate/
sample handling, inconsistent trace rejection and inclusive-loop double counting.
Fresh full regression: **295 passed in89.23s**. Deterministic `--check`, all241
raw-file hashes and all8,659 CSV cells pass. Scoped Git attributes preserve raw
bytes and generated hashes across Windows checkout. Local verification is
publication/protocol evidence, not additional TPU runs.

Recommended next experiment: retain banked embedding + unchanged JAX network,
measure1/8-device execution and actual128-chunk scan with exact-Q checks. Any
prepacking of phase tables must be a separately declared runtime contract;
current numbers include preparation on every call. Dense optimization should
first preserve the observed producer/consumer arithmetic boundary, and LN
requires finer arithmetic evidence. More tiling alone has not solved either.

This report closes the execution-boundary bundle. Its monitor is retired after
the scoped publication; no new session, production default or BN change is
part of this result.
