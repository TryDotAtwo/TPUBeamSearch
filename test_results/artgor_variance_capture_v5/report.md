# Variance capture v5: instrumentation changes the mean/output; attribution rejected

Kaggle completed source `04f9d6b38111fc164fa72f280123a88930b26eb0`, launcher
`48886af`. JSON status is complete, all ten reduction cases finished and all
comparisons are finite. Runtime, device inventory and input/checkpoint/model/
puzzle hashes match v4. Untouched large output hashes reproduce the prior run.

The new JAX variance capture **fails the untouched-output control**:

| Control | 16K/device mismatches | chunk256/device mismatches |
|---|---:|---:|
| New captured output vs untouched JAX | 1,418,986 | 1,418,932 |
| New captured output vs concurrent v4 capture | 1,418,986 | 1,418,932 |
| New captured invstd vs concurrent v4 capture | 0 | 0 |
| Pallas native/split controls | 0 | 0 |

These output differences are numerical, not signed zero. The v4 invstd SHA
reproduces at both shapes. Large captured BF16 Dense remains exact, but the
captured mean differs from Pallas in **28,787,712 broadcast elements**, versus
only 1024 in the previously validated capture. Thus unchanged invstd alone
does not validate the new capture's Dense/mean/output tuple.

## Compiler evidence

In `jax_v4_control_16384.compiled.txt`, the Dense+bias fusion returns both the
BF16 Dense and a sum reduced from its FP32 bias-add intermediate (lines129-145).
The new `jax_capture_16384.compiled.txt` has a separate mean reduction consuming
a BF16 matrix parameter (lines148-152). This is a material change of producer
boundary, consistent with changed mean and output. It does not authorize
identifying every physical rounding from conversion syntax alone.

## Reduction/replay results are not an original-prefix fix

All five Pallas reduction orders have 2048 invstd differences at 16K and zero
at chunk256. Native pair capture and FP32 replay zero-controls pass. However,
these computations consumed the **new, perturbed captured mean**. They are
valid replay measurements on those inputs, not controlled attribution to the
original JAX prefix.

BF16 variance replay differs from captured invstd in about 16.69 million
broadcast elements. At 16K, both JAX and Pallas replay of captured JAX variance
give 16,686,080 differences. This strongly warns against replacing internal
variance with a materialized BF16 buffer, but the failed output control prevents
declaring a production fix or concluding reduction versus rsqrt causality.

## Next experiment

Keep the validated v4 capture completely unchanged. Use **its** Dense, mean,
invstd and output as the fixed inputs/oracles for all Pallas variance and replay
cases. Keep any additional JAX variance capture separate and diagnostic-only;
compare each shared captured field explicitly. Do not feed its changed mean
into the original-prefix experiment. Require reproduction of v4 mean/invstd
hashes and the known zero-output-error JAX-statistics/Pallas-affine control.

Then compare Pallas FP32 variance reduction/rsqrt candidates against v4 invstd
and full prefix output, with native reconstruction controls. Save scalar
variance/mean/invstd bits and bounded affected-row examples; exhaustive expanded
broadcast mismatch coordinates caused unnecessarily large artifacts in v5.

## Artifact storage status

The first full download failed with `No space left on device` on C:. Nothing
was deleted. JSON, the complete Kaggle log and four relevant compiled HLO files
are saved/published here. The large/partial NPZ files on C: are deliberately
excluded from Git (one exceeds GitHub's ordinary per-file limit).

The interrupted D: download was resumed after confirming the old process no
longer existed. Full archive now verified at
`D:/TPUBeamSearchArtifacts/artgor_variance_capture_v5`: 226 files, 1,293,417,572
bytes, 111 NPZ CRC checks passed, no empty files. `archive_manifest.json`
records all file lengths and SHA-256. Large NPZ files remain outside Git;
the remote v5 remains available. No unrelated files were deleted.

No production defaults, BN or beam changed. No full-model exactness or speed
claim. The large-shape JAX oracle is unchanged.
