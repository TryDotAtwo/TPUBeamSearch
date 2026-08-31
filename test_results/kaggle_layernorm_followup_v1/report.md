# TPU arithmetic follow-up v1: results

2026-08-31. Kaggle kernel **COMPLETE**, benchmark JSON **complete**.
This report supersedes the pending status in [launch.md](launch.md).

## Conclusion

The mask compiler failure is now isolated and several workarounds execute on
the actual TPU. **None of the six Pallas-containing full-model configurations
passes the unchanged exact-Q gate on either 16K corpus.** All are slower than
the original JAX in every paired measured round. There is no accepted Pallas
speedup, 32K confirmation, 8-device scaling result or distributed-beam result.

The profiles establish a more useful next target: the common embedding gather
and flattening reshape consume about **5.47ms, approximately half of the JAX
device module time**. They also show that expensive masked Pallas LN execution,
not Python dispatch alone, explains large candidate slowdowns. This is an
attribution result, not proof that those 5.47ms can all be eliminated.

## Provenance and fixed contract

- Kernel: [trydotatwo/tpu-layernorm-arithmetic-followup](https://www.kaggle.com/code/trydotatwo/tpu-layernorm-arithmetic-followup), version1.
- Source: `d58cf9fd8e86ec145c6bbc4f6c7f5aff489d6e21`; launcher: `d87fa2b16d5bc3489d914939db0ce4ba7766b397`.
- Actual hardware: eight visible **TPU v5 lite** devices, **one active**. The requested `v3-8` label is not the executing generation.
- Python3.12.13; JAX/jaxlib0.10.2; libtpu0.0.42.1; NumPy2.5.0; torch2.8.0+cpu; default matmul precision unset, x64 disabled.
- State150, categories150, embedding24, flattened3600, hidden1024, ten two-Dense residual blocks, LN epsilon1e-5, Q30; minimize Q.
- Full candidates keep the JAX embedding/input Dense/LN and output head. Pallas replacements affect residual operators only: twenty Dense and/or twenty LN calls, not30 independent child forwards.
- Original JAX receives runtime FP32 parameter tensors and computes BF16. Typed JAX and candidates receive runtime BF16 tensors. Captured parameters have an explicitly separate control.

All checkpoint/model-source/puzzle/input hashes match
[arithmetic v1](../kaggle_layernorm_arithmetic_v1/report.md):

| Artifact | SHA256 |
|---|---|
| Checkpoint | `2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2` |
| Original jax_model.py | `6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e` |
| Puzzle | `01c616cb943d574d1b63109f350b30c7710656e53e2a5eaebb5d50ed0e495ff0` |
| Legal32768 input | `b7ff6754f3cfbd0c531d14e00b7f516adba8f412c3ee9d93ede5464cc0730fc1` |
| Stress32768 input | `610be0697c4a087f2a66dbcd5622bfdcc13e0fac9efc0106a3d62fa28989ea5d` |

Original checkpoint/source: [Artgor cube555 artifacts](https://www.kaggle.com/datasets/artgor/cube555-tpu-artifacts).
The complete raw record is [stream1_layernorm_followup.json](arithmetic_followup/stream1_layernorm_followup.json).

## Coverage and failure accounting

| Section | Requested | Successful executions | Compilation errors |
|---|---:|---:|---:|
| Synthetic predicate/LN |56|42|14|
| Checkpoint Dense/LN |36|26|10|
| Block4096 |14|14|0|
| Same-suffix controls |2|2|0|
| Full16384 |14|14|0|
| Full baselines |6|6|0|
| Promotion32768 |0|0|0|

The24 errors are Mosaic predicate-layout rejections, not measured execution
latencies or VMEM overflow. Successful comparisons are finite. All eight
timing groups are valid paired comparisons: twelve alternating forward/reverse
rounds, no timing-error fallback. Failed compile rows have no execution timing.
Different case sets mean24 versus the earlier52 errors is not a controlled
failure-rate improvement metric.

`q.*.eligible=true` in the shared metric helper means finite tensors only;
it does **not** mean exact-Q acceptance. All eligible candidate speedup fields
remain null. The original, typed, captured and full-builder JAX controls match
exactly on both full16K corpora; only the two JAX full-builder rows pass among
the14 full rows.

## 1. The predicate failure has a small target reproducer

Both BM128/256 and logical widths1024/130 were exercised on matched synthetic
inputs. Minimal predicates keep even columns and zero odd columns, so they
cannot disappear as an all-valid constant mask.

| Minimal selection | Width/BM cases | Outcome |
|---|---:|---|
| BF16 operands, rank1 boolean broadcast |4|All fail compilation|
| BF16 operands, direct2D predicate |4|All compile and match exactly|
| FP32 operands, rank1 boolean broadcast |4|All compile and match exactly|
| FP32 operands, direct2D predicate |4|All compile and match exactly|

For legacy BF16 LN at width1024, enabling only the input, centered-value or
output mask separately is enough to reproduce the error. The original all-mask
kernel also fails. Removing redundant masks, promoting `where` operands to
FP32 then converting back, or constructing a direct2D predicate all compile.
At width130, only complete population/output masking is used; direct2D and
FP32 selection compile, while the original broadcast selection fails.

All mixed-LN arms compile, but none exactly matches JAX LN. A compilation
workaround is not a numerical fix. Decoded payloads show the intended predicate
construction difference; these embedded StableMosaic payloads precede final
layout passes and are not TPU machine-code dumps.

## 2. Mixed arithmetic improves isolated LN, but is insufficient

Block1 Dense output is held fixed for each corpus,4096x1024 elements:

| Corpus | LN arithmetic | Exact fraction | Mean abs | Max abs |
|---|---|---:|---:|---:|
| Legal | Legacy surviving arms |0.4798133373|0.0007401236074|0.046875|
| Legal | HLO-informed mixed |0.9007606506|0.0001981860953|0.046875|
| Stress | Legacy surviving arms |0.5274770260|0.0007996904925|0.0625|
| Stress | HLO-informed mixed |0.8797957897|0.0002349603575|0.0625|

The mixed expression preserves BF16 mean/variance/inverse boundaries around
FP32 centered values, reductions and affine operations, with BF16-rounded
epsilon. Those coarse boundaries reflect the saved JAX HLO, but matching them
does not establish identical target reduction, instruction or rounding behavior.
The exact remaining arithmetic mechanism is not established by this run.

Legacy `none/fp32_where/direct2d` have equal recorded aggregate error metrics;
the mixed arms likewise have equal aggregates. Pairwise output tensors were
not retained, so those facts do not prove pairwise equality. Source explicitly
makes mixed `all/fp32_where` arithmetically equivalent because operands are
already FP32; those are not independent arithmetic hypotheses.

Synthetic mixed exact fractions are 0.9641075134 at width1024 and
0.9474158654 at width130;
both fail exactness. Padding correctness in CPU tests does not substitute for
these actual TPU results.

## 3. Late Dense: nearly exact locally, rejected in the full model

Standalone late Dense differs in20 legal and31 stress elements out of4,194,304;
max error0.25, mean errors1.1920929e-6/1.8477440e-6. Replacing the first block's
Dense calls retains same-suffix Q exact fractions0.9880859375/0.9850341797.
The legal topK set overlap is1, but order agreement is only0.9140625.

The complete late-Dense/JAX-LN hybrid is materially different from original Q:

| Corpus | Q exact | Max abs | Mean abs | RMSE | Cosine | Argmin agreement | Global topK overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legal |0.3286905924|2.75|0.1034673722|0.1920314775|0.9999735390|0.9265136719|0.7506103516|
| Stress |0.1759155273|1.0625|0.1176228205|0.1531214175|0.9999246509|0.8674316406|0.9298706055|

Tiny standalone disagreement is therefore not an acceptance criterion. This
does not prove one specific accumulation/amplification mechanism: compiler
boundaries and arithmetic require controlled attribution, not subtraction of
aggregate errors.

Same-suffix zero-replacement controls are exact. Separately composed **all-JAX**
prefix/block/suffix already differs from monolithic Q: mean errors
0.1488190107/0.1685348511 and topK overlap0.7502441406/0.9074707031.
Candidate same-suffix errors and this JAX boundary effect remain separate.

## 4. Full-model timing: no accepted candidate, no dispatch-only explanation

Synchronized device-resident calls, no transfers/compilation in the interval.
Five warmups, twelve paired rounds. Values below are diagnostic medians in ms;
every Pallas-containing row fails exact-Q acceptance.

| Full implementation | Legal sync | Legal queued/call | Stress sync | Stress queued/call | Exact Q on both |
|---|---:|---:|---:|---:|---|
| Original runtime JAX |11.494|11.050|11.504|11.051|Yes, oracle|
| Typed runtime JAX |11.422|11.014|11.457|11.023|Yes|
| Captured-parameter control |11.391|11.057|11.443|11.056|Yes, control|
| JAX full-builder control |11.433|11.016|11.480|11.024|Yes, control|
| Late Dense + JAX LN |16.068|15.603|16.088|15.633|No|
| JAX Dense + legacy unmasked LN |12.582|12.159|12.648|12.169|No|
| JAX Dense + legacy FP32-where LN |27.003|26.528|27.016|26.533|No|
| Late Dense + legacy unmasked LN |16.810|16.348|16.822|16.369|No|
| JAX Dense + mixed masked LN |21.445|20.980|21.443|21.013|No|
| Late Dense + mixed masked LN |26.289|25.824|26.291|25.840|No|

Full exact values/samples/spreads are retained in the raw JSON and generated
[CSV summaries](full_summary.csv). On legal inputs, synchronous original JAX
processes about 1.425M states/s, typed JAX 1.434M, and late/JAX 1.020M. The latter
is diagnostic throughput of a rejected implementation, not an eligible speedup.
Each non-control candidate loses to original JAX in all12 paired
rounds on both corpora. Even the closest candidate has positive paired deltas
in every round, rather than a timing-noise tie.

The queue retains and synchronizes all eight outputs of the same executable.
It lowers observed full-model per-call cost by about0.33–0.48ms; the multi-ms
candidate penalties remain. This is amortized queued throughput, not single-call
latency, a real128-chunk scan, or a complete beam depth. Queued microoperators
and blocks are also recorded separately from their synchronized measurements.

## 5. Profiles locate device costs

All ten legal16K baselines/candidates have actual gzip JSON traces and XPlane
protobufs. Each trace contains three executions on TPU0's XLA Modules lane,
with matching nonoverlapping XLA Ops; nested host events are not summed as
device work. Profiling is outside the benchmark timing intervals and remains
diagnostic for rejected candidates.
Device attribution uses the decoded Chrome traces; XPlane protobufs are retained
and hashed but were not decoded. See [profile_summary.json](profile_summary.json)
for every original device-operation name, sample and explicit grouping, and
[profile_summary.csv](profile_summary.csv) for the compact view.

Device module medians, ms: original10.9706, typed10.9416, graph10.9398,
late/JAX15.5331, legacy-unmasked12.0848, legacy-FP32-where26.4480,
late/legacy16.2789, mixed20.9155, late/mixed25.7535. Captured JAX10.9795
does not retain its small synchronized-call advantage in device timing.

Per forward, the common embedding `gather_fusion` costs about4.2586ms and
flattening `reshape`1.207–1.217ms. The captured compiled HLO maps the operation
to embedding[150,24] gathered by states, producing[16384,150,24] and then
flattening to[16384,3600]. This common input work should not be blamed on LN.

Summed residual Pallas LN device time is about1.984ms for JAX-Dense/unmasked,
16.350ms for JAX-Dense/FP32-where, and10.813ms for JAX-Dense/mixed masked LN.
The FP32-where versus unmasked module difference (~14.36ms) is almost entirely
the LN difference; the mixed versus unmasked difference (~8.83ms) behaves
similarly. Late Pallas Dense calls consume about8.34–8.37ms across twenty calls.

Embedded HLO configurations report mixed all/center/FP32-where scoped VMEM
5,554,176bytes versus1,495,040 for none/input/direct2D, and3,108,864 for output.
This is a compiler-scoped configuration, not physical VMEM capacity or proof
that allocation alone causes the timing difference. Direct2D/unmasked mixed LN
were tested as operators, **not** as full-model candidates in this bundle.

## 6. Ranking, masks and input coverage

Legal16K contains11,401 unique states and11,606 distinct(state,last_move) pairs;
categorical stress has16,384/16,384. Neither corpus was deduplicated or replaced.
Legal duplicates beyond the first representative are4,983/16,384 (~30.41%).
Random-walk length is not true puzzle distance; these are not recorded frontiers.

Unmasked Q has491,520 valid parent/action candidates, K16,384. Legal inverse-mask
diagnostics retain476,957; stress retains491,520 because last moves are all-1.
The stress masked metrics are consequently not independent pruning evidence.
All selected-invalid and all-masked-row counts are0.

Oracle K/K+1 gaps are0: legal5337 tied candidates at1.9609375; masked legal8039
at1.96875; stress1337 at9.125. Stable score/flat-ID ties and ordering matter.
For late/JAX legal masked Q, topK overlap drops to0.7322998047 and argmin
agreement to0.8410644531. Global topK remains a proxy: owner quotas, packed
score semantics, receiver dedup/history and replay are not exercised.

## Decision and next discriminating work

Keep the original JAX full model as the accepted implementation. No production
BN/LN defaults were changed. Do not promote to32K or8TPU using these results.

The most concrete next experiment is an exact-value flat embedding gather
that avoids the costly intermediate layout/reshape, first compared alone and
then inside the complete JAX model with the same runtime parameters and exact
Q gate. It is a proposal, not a claimed speedup or an already launched job.
For LN, keep the minimal predicate reproducer and faster direct2D/unmasked
operator arms, but establish remaining arithmetic equivalence before treating
them as full-model winners. Full mixed-direct2D timing remains unmeasured.

This analysis used the TPU coding skill's separate numerical/execution contracts
and two independent numerical/performance audits. Raw evidence, not expert or
reviewer agreement, determines the acceptance decision. No additional TPU job
is submitted by this report.

## Published evidence and verification

All 258 output-file paths were compared with both pages of the Kaggle output
listing: no missing, extra or empty files. The full Kaggle log is stored
separately as [tpu-layernorm-arithmetic-followup.log](tpu-layernorm-arithmetic-followup.log);
the benchmark's own [console log](arithmetic_followup/benchmark.log) is also retained.

Machine-readable views preserve full numerical precision and point back to raw
section/index pairs, including rejected cases:

- [summary.json](summary.json), [full](full_summary.csv), [block](screen_summary.csv),
  [checkpoint operators](operator_summary.csv), [synthetic probes](synthetic_summary.csv)
  and [same-suffix controls](control_summary.csv).
- [HLO directory](arithmetic_followup/hlo/), [profiles](arithmetic_followup/profiles/)
  and [artifact manifest](artifact_manifest.json) with SHA256 and byte counts.

The two captured-control StableHLO files are byte-identical and each exceeds
GitHub's individual-file limit. Their unmodified originals remain local; the
published manifest maps both paths to one shared set of three ordered gzip parts,
each below GitHub's size limit. No HLO content is discarded. The manifest checker
validates the reconstructed byte-stream hash even
in a fresh checkout without those two raw files. Raw evidence is marked `-text`
so Git does not normalize its newlines.

The manifest covers 259 raw files / 422,366,234 bytes. The shared gzip parts
are 36,054,812, 35,855,580 and 34,219,464 bytes; published raw/archived evidence
occupies 132,213,002 bytes before reports and derived summaries. These are local
downloaded-byte hashes and sizes, not a claim that Kaggle supplied remote hashes.

Reproduce from the repository root:

```text
python test_results/kaggle_layernorm_followup_v1/build_summary.py
python test_results/kaggle_layernorm_followup_v1/build_summary.py --self-test
python test_results/kaggle_layernorm_followup_v1/profile_summary.py --check
python test_results/kaggle_layernorm_followup_v1/profile_summary.py --self-test
python test_results/kaggle_layernorm_followup_v1/package_artifacts.py --check --expected-raw-count 259
python test_results/kaggle_layernorm_followup_v1/package_artifacts.py --self-test
```

The numerical audit independently checked 7,600 raw-derived CSV cells and
rejected eight deliberately corrupted in-memory reports. The profile checker
confirms ten profiles / thirty TPU0 module executions, nonoverlapping device
operations, and picosecond-duration consistency. Fresh repository regression:
`python -m pytest -q` -> **252 passed in 44.92s**. These local checks validate
analysis/code integrity; the downloaded target artifacts supply TPU evidence.
The profile parser's three focused tests and packaging's four round-trip,
missing-original, empty-input and no-overwrite tests also pass.
Publication checks caught Windows newline translation in the summary writer;
an LF-only regression failed before the writer fix and passed after it. Derived
values did not change. Helper-script bytes are also protected from checkout
newline conversion because summary provenance hashes the generator itself.
