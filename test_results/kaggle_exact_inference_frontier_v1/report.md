# Exact eight-device LayerNorm inference frontier

Date: 2026-09-01

Kaggle kernel: `trydotatwo/tpu-exact-inference-frontier`, version 1

Launcher commit: `f3273920f8ef38baa6daa9eac12338e1c0b22fcc`

Benchmark source commit: `fc5c87ae5c49c0a92d4ccd634831e8980a7f44e8`

## Result

The kernel completed successfully on eight active `TPU v5 lite` devices. The
frozen gate selected
`exact_split_bm4096_pallas_head_bm256_bk1024_bn128_late`. It is elementwise
and hash exact against the original Artgor Q model on both legal scrambles and
categorical stress, at both the 16K/device screen and the actual 32K/device
confirmation batch.

| Phase | Local / global batch | Corpus | Original JAX, ms | Typed JAX, ms | Accepted exact split, ms | Selected, ms | vs accepted | vs original | Selected states/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| screen | 16,384 / 131,072 | legal | 11.983315 | 11.874006 | 7.284250 | 7.133326 | 1.02116x | 1.67991x | 18.3746M |
| screen | 16,384 / 131,072 | stress | 11.903051 | 11.886271 | 7.229865 | 7.146080 | 1.01172x | 1.66568x | 18.3418M |
| confirmation | 32,768 / 262,144 | legal | 24.865499 | 24.803759 | 15.566380 | 15.363139 | 1.01323x | 1.61852x | 17.0632M |
| confirmation | 32,768 / 262,144 | stress | 24.852184 | 24.793094 | 15.540994 | 15.364385 | 1.01149x | 1.61752x | 17.0618M |

Every selected row has finite outputs, `max_abs = mean_abs = RMSE = 0`,
`exact_fraction = 1`, zero mismatch witnesses and the same output SHA-256 as
its original/control row. The 32K/device confirmation retained the screen
winner; it was not substituted after observing confirmation results.

## What the winner is

The model is the Artgor LayerNorm ResMLP: 150 categorical positions, embedding
`150x24`, hidden width 1024, ten residual blocks, 21 runtime LayerNorms and 30
Q outputs. The winning execution boundary is after the final residual block:

1. a prepacked exact prefix with Pallas flat-embedding custom call and
   `BM=4096`; Dense/LayerNorm/residual work inside this prefix is still
   JAX/XLA-lowered;
2. a separate Pallas Dense head with `BM=256`, `BK=1024`, `BN=128` and late
   BF16 rounding;
3. two device-resident `shard_map` dispatches, with no host materialization of
   the hidden tensor.

This is therefore an exact hybrid JAX/XLA+Pallas inference engine, not yet an
all-Pallas rewrite. It does establish that custom Pallas boundaries can beat
the original monolithic JAX model without changing a single BF16 output.

Most of the gain is from the exact split and the `BM=4096` prefix. At 32K/device
the same prefix with a JAX head measured 15.384900 ms legal and 15.415145 ms
stress. Replacing only that head with Pallas reduced the composed medians to
15.363139 and 15.364385 ms (about 0.14% and 0.33%). The standalone 32K Pallas
head was about 4-5% slower than the standalone JAX head, so the tiny composed
head delta is not a causal proof that the Pallas head itself is universally
faster. The robust claim is the frozen full-pipeline winner against the accepted
`BM=2048` exact control and the much larger win against monolithic JAX.

## Arithmetic and tiling evidence

- Forty Pallas head configurations were screened: five BM values, four BK
  values and two rounding boundaries. Only ten were exact on both corpora:
  late rounding with `BK=128` or `BK=1024`, for every screened BM.
- Late `BK=256` produced 25-34 mismatches and late `BK=512` produced 24-32.
  Rounding before bias produced 851,898-940,235 mismatches. Timing never
  overrode these rejections.
- `prefix_bm2048` and `prefix_bm4096` were exact at both batches. At
  32K/device `BM=4096` measured 15.183760 ms legal and 15.278214 ms stress for
  the prefix alone.
- `prefix_bm8192` and `prefix_bm16384` were compile rejections, not slow
  candidates. Scoped VMEM requests were 19.84 MiB and 24.25 MiB against the
  compiler's 16.00 MiB limit.
- The three materialized-identity experiments were rejected: each had 427
  legal and 663 stress BF16 mismatches. An apparently identity boundary can
  therefore change exact arithmetic.

At 32K/device the compiler's static memory report (not a hardware counter)
gave the original/typed monolithic executables about 1.510 GB temporary memory.
The selected prefix reported 251,946,496 temporary bytes and 67,108,864 output
bytes; its Pallas head reported zero temporary bytes. The saved compiled HLO
shows the Pallas head as `stream1_dense_linear` with `tpu_custom_call`, whereas
the JAX head is lowered as a convolution plus output fusion.

## Protocol and provenance

- Runtime: Python 3.12.13, JAX/jaxlib 0.10.2, libtpu 0.0.42.1, NumPy 2.5.0.
- One process, eight local and active TPU devices.
- Five warmups and twelve synchronized samples per timing group, with
  alternating forward/reverse execution order.
- Checkpoint SHA-256:
  `2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2`.
- Original-model source SHA-256:
  `6d00da89ce45cf84167db20780e30f676cde3ae756d376c8e05a7e0dcf98e46e`.
- Puzzle SHA-256:
  `01c616cb943d574d1b63109f350b30c7710656e53e2a5eaebb5d50ed0e495ff0`.
- Legal/stress input SHA-256:
  `340cce563b0b84eff52fc13da7979be261297e948223db5830fb0c7df267a743`
  and
  `8d054a06a9fcfa339cc77cecab7b98546f5212462c009403955ee390bff53955`.

The immutable JSON contains 86 head rows, eight prefix rows, 19 full
configurations, 38 full rows, twelve timing groups and sixteen declared
diagnostic profiles. `analyze.py` independently recomputes the terminal gate;
its tests also reject a synthetic winner mismatch.

## Profile and Kaggle API incident

The result JSON, full Kaggle log and 110 compiled/stable HLO files were
recovered. Kaggle CLI 2.2.2 intermittently exposed the output list, but its
`ListKernelSessionOutput`, `ListKernelFiles` and status calls then returned
`403 Permission 'kernels.get' was denied` for this newest private kernel. The
same credentials successfully returned account quota, the owner's kernel list
and older private-kernel status. The owner list and full event log both prove
that this exact kernel exists and completed; restarting it would have created
an unnecessary duplicate TPU run.

Only one of the sixteen declared diagnostic trace/XPlane pairs was recoverable
before the endpoint became consistently forbidden. Thirty bounded read-only
retries did not restore access. The available artifact is the 32K/device
accepted-control composed runner. The corrected multi-device analyzer verifies:

- three composed forwards and six compiled dispatches;
- composed device-module median 14.620128 ms;
- TPU:0 XLA-ops sum 14.448927 ms per forward;
- mean prefix-to-head device gap 0.009730 ms;
- 1,317 device-op events, with no lane overlap or ambiguous module ownership.

That profile is diagnostic only and was not used as an acceptance timing. A
winner profile was not recovered, so this report does not invent a
profile-based causal attribution for the winner. The practical fallback for
this Kaggle failure mode is: verify quota/auth, query the owner's kernel list
and event log, request output with `--page-size 200`, use bounded read-only
retries and never restart a terminal kernel merely because one read endpoint
maps every 401/403 to the CLI's misleading wrong-slug message.

## Decision

Adopt the selected exact split as the current eight-device inference frontier
for this checkpoint and these two input corpora. Preserve the original JAX
model as the correctness oracle. Future all-Pallas work should first reproduce
the exact JAX rounding boundaries of each residual Dense/LayerNorm block, then
prove a full-model win at 32K/device; isolated-kernel speed is not sufficient.

Post-download verification passed: six focused artifact/analyzer tests, twenty
TPU-plugin package tests and the complete project suite (`345 passed in
110.77s`).
