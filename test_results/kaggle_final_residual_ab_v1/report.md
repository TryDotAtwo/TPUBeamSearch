# Exact eight-TPU final-residual A/B v1

## Result

The frozen goal is achieved.  `pallas_split_after_final_block` is finite and
elementwise identical to the original BF16 `jax_model.apply` on both legal
scrambles and categorical stress, while beating the fastest exact JAX control
on all eight TPU cores at both measured batch sizes.

The winning inference path is:

1. one compiled `shard_map` dispatch performs the prepacked Pallas banked
   embedding lookup and the unchanged JAX input Dense/LayerNorm plus all ten
   residual blocks;
2. the resulting BF16 `[local_batch, 1024]` hidden matrix stays device-resident;
3. a second compiled `shard_map` dispatch applies the unchanged JAX 1024x30
   output head.

This is full 150x24 embedding -> 3600x1024 Dense -> ten 1024-wide LayerNorm
residual blocks -> 30-score Q inference.  It contains no beam-search work.

| Local batch/core | Corpus | Fastest exact JAX | JAX ms | Winner ms | Global Mstate/s | Speedup |
|---:|---|---|---:|---:|---:|---:|
| 16,384 | legal | `typed_split_before_final_block` | 11.9207 | 7.3210 | 17.904 | **1.628x** |
| 16,384 | stress | `typed_monolithic` | 11.9285 | 7.3123 | 17.925 | **1.631x** |
| 32,768 | legal | `typed_monolithic` | 24.7687 | 15.6471 | 16.753 | **1.583x** |
| 32,768 | stress | `typed_monolithic` | 24.8720 | 15.6529 | 16.747 | **1.589x** |

The global batches are 131,072 and 262,144 states respectively.  Against the
original `jax_model.apply` rather than the fastest exact control, the winner is
1.637/1.644x faster at 16K and 1.589/1.600x faster at 32K.

## Correctness gate

For the winner at both batch sizes and on both corpora:

- all Q values are finite;
- BF16 mismatch count is zero over 3,932,160 values at 16K/core and 7,864,320
  values at 32K/core per corpus;
- max absolute error, mean absolute error and RMSE are all zero;
- exact fraction and argmin agreement are both 1.0;
- output hashes equal the corresponding original-JAX hashes.

The larger 32K/core run was actually executed after promotion.  It selected the
same winner.  No acceptance threshold was relaxed and `error_count` is zero.

## Complete 16K/core matrix

Mismatch counts are full-Q element mismatches against the original oracle, not
sampled hidden-state diagnostics.

| Arm | Legal ms | Legal mismatches | Stress ms | Stress mismatches |
|---|---:|---:|---:|---:|
| `original_shard_map` | 11.9843 | 0 | 12.0229 | 0 |
| `typed_monolithic` | 12.0739 | 0 | 11.9285 | 0 |
| `pallas_monolithic` | 7.2192 | 12 | 7.2078 | 55 |
| `pallas_barrier_before_final_block` | 7.2127 | 12 | 7.2841 | 55 |
| `pallas_barrier_before_final_dense2` | 7.2445 | 12 | 7.2978 | 55 |
| `pallas_barrier_after_final_dense2` | 7.2651 | 25,453 | 7.2843 | 30,824 |
| `pallas_barrier_after_final_block` | 7.4440 | 12 | 7.3230 | 55 |
| `pallas_barrier_before_and_after_final_dense2` | 7.2651 | 25,453 | 7.3790 | 30,824 |
| `typed_tap_before_final_dense2` | 12.2437 | 12 | 12.3086 | 55 |
| `pallas_tap_before_final_dense2` | 7.4152 | 12 | 7.5442 | 55 |
| `typed_tap_after_final_dense2` | 12.2261 | 0 | 12.3397 | 0 |
| `pallas_tap_after_final_dense2` | 7.4260 | 12 | 7.5128 | 55 |
| `typed_tap_after_final_block` | 12.1367 | 0 | 12.2530 | 0 |
| `pallas_tap_after_final_block` | 7.3782 | 0 | 7.5283 | 0 |
| `typed_split_before_final_block` | 11.9207 | 0 | 12.0162 | 54 |
| `pallas_split_before_final_block` | 7.2317 | 22 | 7.3040 | 54 |
| `typed_split_before_final_dense2` | 12.1133 | 12 | 12.2149 | 55 |
| `pallas_split_before_final_dense2` | 7.4493 | 12 | 7.4508 | 55 |
| `typed_split_after_final_dense2` | 12.1892 | 25,474 | 12.1054 | 30,854 |
| `pallas_split_after_final_dense2` | 7.4087 | 25,474 | 7.4482 | 30,854 |
| `typed_split_after_final_block` | 11.9979 | 0 | 12.0456 | 0 |
| `pallas_split_after_final_block` | **7.3210** | **0** | **7.3123** | **0** |

The matched one-dispatch tap after the complete final block is also exact.  It
is slower and more variable than the real two-dispatch winner because the timed
call must return and synchronize both Q and the 32 MiB/core hidden tap.  None of
the five optimization-barrier arms repairs the monolithic candidate.  Cutting
inside the final block changes BF16 materialization and is not semantically
neutral, as the large `after_final_dense2` mismatch controls show.

## Why the split fixes the result

Compiled HLO closes the attribution left by execution A/B v1.  There are 22
MXU Dense schedules in the monolith.  Convolution index 20 is the second Dense
of residual block 9 and is the only schedule that differs between the exact
typed-JAX and fast Pallas-embedding monoliths:

| Executable | Final residual Dense2 iteration bounds | Result |
|---|---|---|
| typed-JAX monolith | `2x16x1` | exact |
| Pallas monolith | `1x22x1` | 12/55 mismatches |
| every Pallas barrier arm | `1x22x1` | inexact |
| Pallas tap after final block | `2x16x1` | exact |
| winner prefix through final block | `2x16x1` | exact |

The winning prefix contains 21 MXU schedules and ends with the canonical
`2x16x1` schedule.  Its separately compiled head contains one MXU operation,
`EmitOutputBatchInLanesInputBatchInSublanes` with bounds `1x3x1`.  The typed and
Pallas split heads have identical compiled-HLO hashes.  Thus the real boundary
does not repair wrong embedding values: it prevents the fast embedding graph
from perturbing the final residual Dense lowering.  Previous split controls
already established that the banked embedding produces the exact BF16 encoded
tensor.

This is strong compiler-level attribution for this runtime and shape.  It is
not a claim that `1x22x1` is universally inaccurate or that TPU hardware itself
is nondeterministic; the difference is graph- and compiler-schedule-dependent.

## Compiler estimates

At 16K/core, compiler-reported static storage is:

| Executable | Arguments | Output | Temporary |
|---|---:|---:|---:|
| typed-JAX monolith | 52,013,568 B | 1,048,576 B | 758,779,904 B |
| winner prefix | 52,328,448 B | 33,554,432 B | 129,827,840 B |
| winner head | 33,620,480 B | 1,048,576 B | 0 B |

The 32 MiB prefix output is exactly one local BF16 16,384x1024 hidden matrix.
It remains on TPU and is consumed by the head dispatch; no host conversion is
inside the timing.  These are compiler allocation estimates, not HBM traffic,
cache-residency measurements or hardware counters.

## Reproducibility

- Private Kaggle kernel: `trydotatwo/tpu-final-residual-ab`, version 1.
- Public benchmark source: `267df37cd3a35b19ad6250d43768bfd5b536b67c`.
- Launcher commit: `2b56e64`.
- Python 3.12.13, JAX/jaxlib 0.10.2, libtpu 0.0.42.1, NumPy 2.5.0.
- Eight active `TPU v5 lite` devices in one process.
- Checkpoint SHA-256:
  `2b540c3e396f7fb5710ccc44201a698740df1761495ee4059be706374e8e5ac2`.
- Model-source, puzzle and input hashes match the preceding execution A/B run
  and are retained in the JSON.
- Five warmups and twelve paired, alternating forward/reverse, synchronized
  samples per corpus and batch size.  Compilation, first execution, host
  conversion and one-time bank preparation are excluded.
- Raw JSON, benchmark log, full Kaggle log and all 152 StableHLO/compiled-HLO
  files are retained beside this report.  A secret-pattern scan is clean.

The measured winner has also been promoted into
`tpu_beam_search.stream1_layernorm_exact` as an opt-in reusable two-stage API.
Its tests compare the production formulas directly with the benchmarked
`after_final_block` implementation and ensure that the sharded wrapper keeps a
real device-resident dispatch boundary.  The existing BN path and prior
inference defaults are unchanged.  Fresh local regression: 337 tests passed in
131.44 seconds.
