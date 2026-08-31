# LayerNorm arithmetic A/B: controlled optimization experiment

Status: implementation and CPU validation; TPU results pending. This is not an
inference speedup report. Source: `benchmarks/stream1_layernorm_arithmetic.py`.

## Fixed scope

Artgor `q555_2k_BEST.pt`, picture cube555, embedding **150 classes x 24 features**,
150 input labels, 3600→1024 input layer, 10 two-Dense residual blocks, LayerNorm
epsilon 1e-5, 30 Q scores per parent. BN code/default behavior is unchanged.
The input path remains embedding gather, the previously measured LN input winner.

The new runner pins JAX/jaxlib 0.10.2 and libtpu 0.0.42.1. The latter matches
JAX 0.10.2's published `tpu` dependency range (`libtpu==0.0.42.*`), verified from
[the package metadata](https://pypi.org/pypi/jax/0.10.2/json) on 2026-08-31.
Historical runners upgraded libtpu without recording its exact version. Therefore
new in-job baselines, not a difference from historical wall times, establish speedup.
Device kind, software versions, actual source SHA and artifact hashes are recorded.

## Arithmetic switches (LN-only, old defaults preserved)

- `dense_rounding="late"`: FP32 dot accumulator + bias, then BF16 output.
- `dense_rounding="bf16_before_bias"`: completed accumulator rounds to BF16
  before bias, then BF16 output. K-tile partial sums do not round independently.
- `mean_mode="sum_div"`: historical sum-then-division expression.
- `mean_mode="jax"`: FP32 reduction **and division before** conversion to BF16,
  matching the `jnp.mean` expression boundary. Applied to mean and variance.
- `fp32_statistics`: independent switch for normalization arithmetic.

New interpreter tests expose actual differences, including a random Dense
bias-rounding witness, width130 LN padded to aligned storage, multiple K tiles,
and the complete network in separate/per-layer/per-block modes. CPU JAX 0.10.1
is not evidence of TPU 0.10.2 lowering or performance. In particular, width130's
mean witness does not establish an error mechanism for power-of-two width1024.

## One bundled job

For each of two corpora:

1. Original source JAX with runtime FP32 parameters, typed JAX with runtime BF16
   parameters, original captured-parameter control, segmented JAX and identical
   JAX-suffix controls at depths 0/1/3/10. Report direct tensor differences.
2. Dense and standalone LN probes on the **same** first/second-sublayer inputs,
   with compiled HLO saved. Different incoming activation is not an operator A/B.
3. 30 block cases at batch4096: 12 independent Dense/LN crosses, six crosses
   confined to the first or second sublayer, and 12 fused cases. Fused cases
   compare BM128 per-layer/per-block on matching BK256/BN512, then BM256 per-layer.
   Known BM256 whole-block VMEM failures are not retried.
4. Eight complete model cases at batch16384: legacy/early separate, early
   per-layer BM128/BM256, early per-block BF16/FP32 statistics, and two hybrids
   (JAX input/head with Pallas Dense or LN in the trunk).
5. At most two full-output-exact candidates are rechecked at batch32768. Selection
   for this check and its outcome are separate fields. Failure at32768 cannot
   inherit acceptance at16384.

Parameters are runtime arrays for candidates and matched baselines. Full-model
timings include both original FP32-parameter source and preconverted BF16 JAX;
ratios against these have different meanings. Compilation, first execution,
warmup and seven synchronized samples are separate. The baseline is retimed
after each full-model candidate. Exact candidates also get a profiler trace.
One local TPU device runs this experiment; this is not 8-device scaling.

## Inputs, selection and acceptance

`puzzle_info.json` is mandatory. Its generator insertion order defines Q columns;
the loader verifies actual permutations, identity central state and named inverse
composition. Legal states are random walks of lengths 0/1/2/4/8/16/32/64/128,
seed42, with immediate inverses allowed. Categorical iid stress uses seed43 and
is explicitly not reachable-state data. Every candidate sees the same corpus
prefix. Walk length is not optimal distance; neither corpus is a real frontier.

Scores are **minimized**, flattened parent-major/move-minor. Record tensor error,
RMSE, cosine, exact fraction, row argmin, auxiliary argmax, global top-K identity
overlap/order, valid counts and tie/margin summaries. Original notebook settings
had `NO_BACKTRACK=False`; unmasked selection is primary and inverse exclusion is
an additional diagnostic. Stress inputs have incoming-move sentinel -1 (no ban).

Global top-K is only a proxy: actual owner quotas, packed scores, receiver hash
deduplication and history are not reproduced. No solve/replay claim follows.
For this diagnostic the conservative optimization eligibility is **exact original
Q output in both corpora**. Approximate candidates retain measurements but cannot
win on speed. This is not a newly invented 99% tolerance or a universal claim
that useful inference must always be bitwise equal. If none pass, use the full
metric curves and HLO to choose the next experiment rather than weaken the gate.

## Failure handling and provenance

Save atomic strict-JSON checkpoints before/after each case. Compiler errors stay
attached to their case. Missing optional captured controls do not cancel runtime
experiments. Failed required baselines prevent acceptance of affected cases;
uncaught errors mark the partial report `error`. Zero successfully executed full
candidates is an error, not a successful optimization.

GitHub remains source of truth. The private launcher checks a full SHA before
running. Keep one TPU session; do not restart QUEUED/RUNNING work. Collect JSON,
full log, HLO and useful profiles after terminal status. Update the experiment
ledger and reusable plugin evidence only with verified outcomes and their scope.
