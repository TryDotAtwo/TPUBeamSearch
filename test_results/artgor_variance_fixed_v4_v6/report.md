# Fixed-v4 variance v6: valid controls, no reduction winner

Private Kaggle `trydotatwo/tpu-artgor-pallas-prefix-gate/6` completed at source
`a0389b3336bd11336a531bfeeecb338641005d2b`, launcher `a41c5f2`.
Eight TPU v5 lite devices, JAX/jaxlib 0.10.2, libtpu 0.0.42.1, x64 enabled.
The runtime helper's `active_device_count=1` is its default inventory field;
the benchmark explicitly constructs an eight-device mesh. Corpus: all 131072
legal42 states, evaluated at 16384/device and reconstructed chunk256/device.
Input, checkpoint, model-source and puzzle SHA match v4.

## Controls and result

`use_v4_inputs=true`. Both shapes have `attribution_controls.valid=true`:
unchanged capture, Pallas native/remainder/split controls, v4 mean/invstd SHA,
and paired variance native reconstruction pass. Fixed JAX Dense/mean/invstd
through the shared Pallas affine reproduces the untouched JAX prefix exactly.
At large shape its full SHA is
`9755606bffa3d179337f5741fcd23dce5f0469d6b11ebc56c546f6e25b6cd7f0`.

All five orders (native, lanes_serial, lanes_tree, tiles_serial, tiles_tree):

| Comparison | 16K/device mismatches | chunk256/device mismatches |
|---|---:|---:|
| Candidate BF16 invstd vs validated JAX | 2048 | 0 |
| Candidate prefix with fixed JAX Dense/mean | 1275 | 0 |
| FP32 variance replay vs native candidate invstd | 0 | 0 |
| JAX statistics through Pallas affine vs original | 0 | 0 |

All five candidates have identical complete BF16 invstd and prefix-output
SHA at each shape. All comparisons are finite. Invstd differences occupy only
rows 760 and 28870 (1024 broadcast elements each), not 2048 distinct states.

| Row | Native FP32 variance | Pallas invstd BF16 bits | JAX bits |
|---|---:|---:|---:|
| 760 | 61.57134246826172 | 15875 | 15874 |
| 28870 | 39.225154876708984 | 15907 | 15908 |

Compared with native, complete scalar FP32 variance differs on 0 / 26702 /
82935 / 62790 rows for lanes_serial / lanes_tree / tiles_serial / tiles_tree.
Thus the reduction variants genuinely change arithmetic, but none changes
the final BF16 invstd. At row760 tiles_tree variance is 61.57135009765625;
row28870 variance is identical for all five orders. This rejects these five
orders as a fix, not every possible reduction/producer mechanism.

## Invalid diagnostic capture remains isolated

`diagnostic_variance_valid=false` at both shapes. The extra five-slot capture
still perturbs the mean/output, although its invstd matches v4. Consequently
its variance, `jax_shape_*` fields, and JAX-variance replay are diagnostic-only;
they are not a validated description of the original variance producer.
Materialized BF16 variance replay has 16686080 invstd differences at large
shape; this does not justify changing production variance rounding.

The validated v4 HLO has FP32 reduction, multiply by 1/1024, a BF16 roundtrip,
FP32 epsilon add and rsqrt (`jax_v4_control_16384.compiled.txt`, lines174-186).
Conversion syntax alone does not prove physical BF16 rounding. Pallas FP32
materialization/replay reconstructs every candidate exactly; no JAX consumer
of that same FP32 buffer was tested yet. Rsqrt versus upstream producer
precision remains unresolved, not established by this experiment.

## Next controlled A/B

1. Keep the unchanged v4 capture and all current controls. Use the complete
   corpus at both shapes, not a two-row fitted correction.
2. Feed the same real Pallas FP32 variance buffers to JAX and Pallas
   epsilon+rsqrt consumers. Compare scalar 1D and broadcast 2D layouts, with
   explicit FP32 arithmetic and original BF16-expression controls separately.
3. Compare both consumers against validated JAX invstd and reconstruct the
   prefix through the same affine. A consumer match does not validate an
   instrumented variance producer; retain the output capture control.
4. Preserve compiled HLO and full scalar bits. If consumers agree but miss
   JAX, investigate centered-square producer precision/reduction boundaries;
   if they disagree, isolate rsqrt/layout lowering before changing kernels.

No full-model, all-Pallas equality, timing, speedup, or default promotion.
The mean discrepancy remains a separate issue. Raw bounded mismatch NPZ are
examples only (at most eight rows); JSON counts/SHA and scalar NPZ are exhaustive.
An expert consultation was blocked before delivery by external-data review;
there is no expert endorsement of these conclusions.
