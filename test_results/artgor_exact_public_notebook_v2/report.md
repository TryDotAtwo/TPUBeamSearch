# Artgor exact public-notebook validation v2

The actual packaged notebook
[`trydotatwo/cayleypy-cube555-tpu-beam-q-exact`](https://www.kaggle.com/code/trydotatwo/cayleypy-cube555-tpu-beam-q-exact)
version 2 completed on eight TPU v5 lite devices.  It loaded public source
commit `2b99bdf5116f828a21d35b2c5910467f6ab039c2` and the pinned runtime
JAX/jaxlib 0.10.2 plus libtpu 0.0.42.1.  The selected engine was
`exact_split`.

## Immutable-run comparison

The notebook ran the same two pids, two frames, checkpoint and global beam
`16,777,216` as the preserved Artgor `scriptVersionId=344319112` run.  All
four records match on pid, frame, inversion, found status, checkpoint and
beam.  Both found paths match the original path hashes and were verified by
the runtime.

| pid | frame | Result | Path length | Original | Exact | Wall speedup |
|---:|---:|---|---:|---:|---:|---:|
| 1034 | 0 | verified solve | 116 | 3346.595 s | 2869.031 s | 1.1665x |
| 1034 | 7 inverse | not found | - | 9339.882 s | 8007.539 s | 1.1664x |
| 1020 | 0 | verified solve | 120 | 3496.465 s | 3002.461 s | 1.1645x |
| 1020 | 7 inverse | not found | - | 9336.046 s | 8049.799 s | 1.1598x |

The four frame-runs take 21,928.830 seconds instead of 25,518.988 seconds,
an aggregate measured wall-speedup of **1.1637x**.  This is a matched
whole-frame solver measurement for these four records.  It does not mean that
the complete beam-search pipeline is 1.5x faster: the previously demonstrated
approximately 1.6x result applies specifically to full-Q model inference.

## Verification boundary

The downloaded results and raw Kaggle log remain local.  Their SHA-256 hashes
are recorded in `summary.json`.  This public directory deliberately excludes
competition input rows, initial states, solution strings, move-index arrays,
`submission.csv`, and the raw private log.
