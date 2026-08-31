# TPU LayerNorm arithmetic A/B v1 — completed, no promoted candidate

2026-08-31. Kernel `trydotatwo/tpu-layernorm-arithmetic-ab` version1 and report
status are COMPLETE. **No full-model Pallas candidate passed the predeclared
exact-original-Q gate.** JAX remains the measured full-model choice; this is
not a claim that future Pallas optimization cannot win.

## Provenance and execution scope

- Benchmark source: `2e9602829b8e4fa8498b64461f64c556e77ad4f4`.
- Actual inventory: **TPU v5 lite**, eight visible devices, one active device,
  one process. The CLI requested `v3-8`; that request did not establish v3 hardware.
- Python3.12.13; JAX/jaxlib0.10.2; libtpu0.0.42.1; NumPy2.5.0; Torch2.8.0+cpu.
- Same Artgor checkpoint: embedding150×24, input3600→1024, ten residual blocks,
  LayerNorm epsilon1e-5, Q head30. The auxiliary V head is not executed.
- Checkpoint SHA256: `2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2`.
- Source `jax_model.py` SHA256: `6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e`;
  independently matches the previously retrieved local source file.
- Puzzle SHA256: `01c616cb943d574d1b63109f350b30c7710656e53e2a5eaebb5d50ed0e495ff0`.
  Checkpoint/puzzle hashes are runtime-reported; those input files were not
  downloaded again for an independent byte check.
- Legal corpus SHA256: `b7ff6754f3cfbd0c531d14e00b7f516adba8f412c3ee9d93ede5464cc0730fc1`.
  Categorical stress: `610be0697c4a087f2a66dbcd5622bfdcc13e0fac9efc0106a3d62fa28989ea5d`.
- Three warmups and seven synchronized device-resident samples per timing;
  compilation and first execution are separate fields. No transfers, distributed
  selection, complete beam depth, multi-device scaling or real128-chunk scan is timed.

The kernel completed in roughly278s including setup and output conversion.
Its COMPLETE status does **not** mean all candidates compiled:

| Stage | Recorded cases | Executed | Compile errors |
|---|---:|---:|---:|
| Baseline controls | 8 | 8 | 0 |
| Dense/LN operators | 28 | 20 | 8 |
| Block screen: 30 cases × 2 corpora | 60 | 28 | 32 |
| Full-model JAX baselines | 4 | 4 | 0 |
| Full candidates: 8 cases × 2 corpora | 16 | 4 | 12 |

Eight additional JAX boundary-control records are comparisons, not failed cases.
All52 rejected cases are Mosaic predicate-relayout errors, not measured slow
executions or VMEM/OOM rejections. Both input corpora reached the full-model stage.

## Full-model performance, batch16384

“Source FP32 parameters” means `original_apply(..., dtype=BF16)` with FP32
parameter storage, **not FP32 network execution**. Typed JAX uses preconverted
BF16 runtime weights. The two produce exactly the same Q outputs on both corpora.

| Implementation | Legal ms | Legal Mstates/s | Stress ms | Stress Mstates/s |
|---|---:|---:|---:|---:|
| Original JAX, FP32 runtime parameters | 11.50652 | 1.42389 | 11.50301 | 1.42432 |
| Typed JAX, BF16 runtime parameters | 11.51167 | 1.42325 | 11.63601 | 1.40804 |
| Hybrid: early Pallas Dense, JAX LN/input/head | 16.32125 | 1.00384 | 16.20227 | 1.01122 |
| Early Pallas Dense + FP32 LN, per-block BM128 | 29.11755 | 0.56268 | 29.28278 | 0.55951 |

Candidate times above are **diagnostic only**, because both fail correctness.
Hybrid sample ranges:16.18954–16.44499ms legal,16.11356–16.35393ms stress.
Per-block:29.06776–29.24309ms legal,29.08715–29.47888ms stress. Paired BF16 JAX
baselines retimed after these cases are11.59954/11.68344ms and11.66679/11.63787ms,
respectively. The hybrid is about1.39–1.41× slower; per-block about2.50–2.52× slower.
Weight preparation does not explain either the numerical difference or this gap.

The six other full configurations—legacy separate, early separate, early
per-layer BM128/BM256, early per-block BF16 LN, and JAX-Dense/Pallas-BF16-LN
hybrid—were compile-rejected in both corpora. No fabricated latency is assigned.

| Executed candidate / corpus | Max abs Q | Mean abs Q | RMSE | Exact Q % | Argmin % | Top-K overlap % |
|---|---:|---:|---:|---:|---:|---:|
| Early Dense/JAX LN, legal | 3.7500 | .217743 | .384465 | 20.7560 | 75.9949 | 61.4502 |
| Early Dense/JAX LN, stress | 1.4375 | .188193 | .239673 | 10.8693 | 79.3579 | 88.3850 |
| Early per-block FP32 LN, legal | 3.5000 | .285102 | .493316 | 19.4712 | 76.3977 | 60.7361 |
| Early per-block FP32 LN, stress | 1.3125 | .153202 | .196068 | 13.4281 | 81.0486 | 90.4053 |

All four outputs are finite. High cosine (.999816–.999896 for the hybrid,
.999829–.999877 per-block) does not establish decision agreement. Legal
inverse-mask top-K overlaps are55.7800% hybrid and53.6682% per-block. The
unmasked top-K order fractions are near zero (.0183–.0305% hybrid,
.0549–.3845% per-block); ordering is tie-sensitive as discussed below.

**No candidate was selected for32768; no32768 timings or profiles exist.**
The exact-at-larger-batch list is empty because no16K candidate passed;
it is not evidence of a separately evaluated32K result.

## JAX compilation boundaries are a real control effect

At batch4096, typed runtime JAX and captured-source JAX both match the source
runtime oracle exactly on both corpora. Their respective legal latencies are
3.05481 and2.98893ms versus3.01750ms original; stress2.98059/2.89458 versus2.97295ms.
Captured source was not remeasured at16K.

Splitting **JAX alone** into a compiled input prefix, individually compiled
blocks, and a jointly compiled suffix changes Q relative to monolithic JAX:

| Corpus | Separately run residual blocks | Q mean abs | Exact Q % | Argmin % | Top-K overlap % |
|---|---:|---:|---:|---:|---:|
| Legal | 0 | .144582 | 27.6921 | 90.8691 | 67.7246 |
| Legal | 1 | .148819 | 26.6154 | 90.7715 | 75.0244 |
| Legal | 3 | .140042 | 27.4089 | 91.1133 | 69.2139 |
| Legal | 10 | .145608 | 25.5404 | 91.1865 | 69.4580 |
| Stress | 0 | .166863 | 12.4829 | 83.0566 | 90.4785 |
| Stress | 1 | .168535 | 12.1981 | 82.7637 | 90.7471 |
| Stress | 3 | .168684 | 12.1525 | 83.6914 | 90.9912 |
| Stress | 10 | .169711 | 12.5146 | 83.2031 | 90.8936 |

Depth0 still has a separate input prefix; it is not the monolithic graph.
Every same-suffix self-control is exact. `cross-jax-jax` also has exact hidden
and same-suffix Q results, while reproducing the depth1 boundary error against
monolithic JAX. Therefore hybrid-versus-monolithic differences cannot be called
pure Pallas perturbation amplification. This confirms a confound in the old
interpretation; it is not a numerical decomposition of the older run, whose
inputs and unpinned libtpu differed or were not fully recorded.

## Dense rounding: the CPU hypothesis did not transfer

Each standalone Dense probe compares4096×1024=4,194,304 output elements.

| Corpus / Dense | JAX ms | Pallas late ms | Late unequal elements | Early unequal fraction |
|---|---:|---:|---:|---:|
| Legal / first | .29192 | .44395 | 20 | 22.8723% |
| Legal / second | .26108 | .45446 | 50 | 25.0825% |
| Stress / first | .30914 | .43038 | 31 | 22.4091% |
| Stress / second | .31171 | .44072 | 59 | 25.5347% |

`late` means FP32 accumulated dot plus bias before BF16 conversion. Its maxima
are.25/.03125/.25/.0625, but only4.77–14.07ppm of elements differ. Forcing BF16
before bias makes the TPU result substantially less like compiled JAX, despite
the source-level/CPU rationale. Keep `late` as the existing default. The cause
of its few remaining disagreements is not isolated by this run.

With JAX LN retained, `cross-late-jax` block latency is.56534/.56723ms versus
.39290/.38721ms JAX:1.44–1.46× slower. Against the **same compiled suffix**, its
legal/stress Q mean errors are only.00410283/.00290680, exact fractions
98.8086/98.5034%, and argmin99.7070/99.7559% (12/10 changed rows of4096).
Top-K set overlap100/99.8779% still differs from order agreement91.4063/18.0176%.
This near match is not exact and is not a full-model result.

The fixed full-case plan did **not** include `late Dense + JAX LN` across the
whole model; that remains an explicit missing experiment, not a failed result.

## LayerNorm and fusion

Both BF16-statistic mean modes (`sum_div`, `jax`) failed TPU compilation; their
intended arithmetic could not be compared. FP32 LN executes, but changes both
statistics and normalization/affine arithmetic versus the BF16 source oracle.
It is not automatically “more correct” relative to that execution contract.

Standalone FP32 Pallas LN takes.37366–.43875ms versus JAX.24359–.29668ms.
Legal first/second LN exact fractions are4.2468/40.1669%, stress8.4373/32.0883%.
With JAX Dense unchanged, the FP32-LN replacement alone gives same-suffix Q mean
errors.191472/.173549 and top-K58.3252/89.2090% legal/stress. That isolates a large
effect in the changed LN path; it does not identify one individual rounding step.

Matched BM128 per-block fusion is modestly faster than per-layer in successful
FP32 cases, but neither wins. Best fused block here is per-layer BM256/BK256/BN512
with `late`: .72423/.74421ms,1.84–1.92× slower than JAX. Within each rounding mode,
fusion variants have identical saved metrics, **not proven pairwise identical
tensors**: their mutual array differences were not recorded. All30 cases remain
listed, including errors, in `screen_summary.csv`.

## Compiler failure and HLO evidence

All52 failures report `MosaicError`, at `broadcast_in_dim`, with the same invalid
predicate relayout.46 involve `vector<128x1024xi1>`, six`vector<256x1024xi1>`:

```text
Non-singleton logical dimension is replicated in destination but not in source
"32,{*,0},(8,128),-2" -> "32,{*,0},(8,128)"
```

The shared suspect is the broadcast boolean mask in `where(valid[None,:], ...)`
within the LN kernels. Standalone LN fails without Dense or pipeline machinery.
The trace does not identify which mask use; no minimal target reproducer or
verified compiler fix was executed. Older runners upgraded libtpu without a
version record, so the package pin alone cannot prove a regression's cause.

In `hlo/legal_scrambles-first-dense-jax.txt`, lines62–68 have a nominal BF16
convolution, FP32 bias addition and BF16 output conversion; line77 wraps these
in one `convolution_add_fusion`. That is not evidence of a separately materialized
pre-bias BF16 value. The observed rounding A/B must take precedence over that
literal reading of an intermediate HLO type.

JAX LN HLO shows a mixed schedule: FP32 reduction then BF16 mean; FP32
centering/square/reduction without explicit intermediate BF16 conversions;
BF16 variance and inverse-standard-deviation boundaries; FP32 affine arithmetic
and final BF16 output. Neither “FP32 everywhere” nor “round after every source
BF16 operation” describes this dump.

Pallas files expose `tpu_custom_call` with a serialized Mosaic body; outer HLO
does not establish machine-level internal rounding. There are20 HLO files,
only for operators that compiled. Failed compiles have logs, not HLO snapshots.

## Ranking and corpus limitations

Legal states are actual generator walks from identity, seed42, requested depths
0/1/2/4/8/16/32/64/128, immediate inverses allowed. Each depth occurs3641 times
in32768 rows except128 occurs3640. At4096 there are456 identical zero-walk solved
states; cancellations and other duplicates are also possible. Length is not
optimal distance. This is not a recorded deduplicated beam frontier.

Minimizing top-K uses parent-major/move-minor IDs, K=batch and stable
`(score, flat_id)` ordering. At16K both reference boundary gaps are zero;
5337 legal and1337 stress candidates tie at K. Legal/stress row best–second
gaps are zero in1424/1423 rows. Overlap and especially ordering depend on these
ties and duplicates; their percentages are not solve-rate estimates.

The notebook's saved `NO_BACKTRACK=False` makes unmasked metrics primary;
inverse exclusion is a separate legal-corpus diagnostic. Stress incoming moves
are -1, so its mask is all-valid. Owner quotas, score packing, receiver dedup,
history and solution replay are absent. `eligible` in the metric helper means
finite comparison only; optimization acceptance is the separate exact-Q flag.

## Decision and next bounded experiment

Do not switch inference to the early-rounding/FP32-LN candidates, do not weaken
the exact-Q gate after these results, and do not launch scaling from them.
BN source/default behavior and unrelated artifacts remain untouched.

Recommended next work, **not executed by this report**:

1. Reproduce standalone BF16 LN failure on one128×1024 tile with this same runtime.
   Test omission of redundant masks only when logical width equals physical width;
   retain masking for padded widths. Reduce predicate/where uses individually if needed.
2. Test an HLO-informed mixed-precision LN against standalone JAX, then a block.
   Separately materialize dot-only output followed by bias to isolate the physical
   rounding boundary. Neither proposal is a verified fix yet.
3. Include full-model `late Dense + JAX LN`, matched typed-JAX timing and the same
   reference-boundary controls. Add a deduplicated/depth-stratified legal corpus
   or actual frontier data before interpreting ranking as search quality.

No new TPU session was launched from the COMPLETE branch. The monitoring
automation is removed after publication, as requested.

## Published artifacts

- `arithmetic_ab/stream1_layernorm_arithmetic.json`: unchanged source result.
- `arithmetic_ab/benchmark.log` and `tpu-layernorm-arithmetic-ab.log`: complete logs.
- `arithmetic_ab/hlo/`:20 compiled operator HLO dumps.
- `controls_summary.csv`, `operator_summary.csv`, `screen_summary.csv`,
  `full_summary.csv`: extracted comparison fields; empty timing cells are rejections.
- `artifact_manifest.json`: byte sizes/SHA256 for all23 downloaded raw files;
  local `.gitattributes` preserves their bytes rather than converting newlines.

Raw logs were checked for common credential/token patterns; none were found.
No checkpoint weights, original input dataset or unrelated local artifacts are
published by this change. Case inventories and CSV values were checked against
the raw JSON, rather than inferred from terminal status or earlier reports.
Publication checks: all 23 raw SHA256/size pairs matched; all 112 CSV rows
(1,648 fields) matched JSON-derived values; `python -m pytest -q` passed
142 tests in 72.07s. Source and BN implementation were unchanged by this report.
