# v10: materialized mean substitution isolates the remaining input error

Source `98de983508534f62726aa1d34d19bed8dd5e4d28`; 256 states/device,
eight TPU v5 lite; runtime, checkpoint, model-source and input hashes match v9.

| Corpus | External BF16 mean source | vs native Pallas | vs JAX input prefix | Same-suffix Q | Original full Q |
|---|---|---:|---:|---:|---:|
| legal42 | JAX | 0 | 0 | 0 | 44125 |
| legal42 | Pallas | 0 | 0 | 0 | 44125 |
| stress43 | JAX | 15 | 0 | 0 | 53852 |
| stress43 | Pallas | 0 | 15 | 17 | 53854 |

The native-Pallas-mean zero-change control is exact on both complete corpora.
The JAX and Pallas BF16 means are identical for every legal row; on stress
they differ only at row 1085: JAX 0.0283203125, Pallas 0.0281982421875. The
reported 1024 mean mismatches reflect broadcasting that one scalar across
the row, not 1024 independent row errors.

Saved mean bits confirm exactly one differing row: JAX `0x3ce8`, Pallas
`0x3ce7`. The same-buffer trace row sums are approximately 28.937508 and
28.937498 respectively, on opposite sides of the rounding threshold 28.9375.
The saved affected raw/reference/candidate rows are finite. Actual BF16-output
HLO keeps the reduction, mean conversion and BF16 output buffer distinct.

Replacing only the materialized mean, with identical raw Dense values and
the same Pallas remainder, removes all 15 input-prefix errors and all 17
same-suffix Q errors. Both legal and stress prefix tensors now match their
JAX reference SHA. This is a controlled causal attribution for these corpora:
the remaining input error can be removed by changing mean computation.
It does not establish the exact physical reduction tree inside JAX, nor
generalize to six corpora, large batches, or the full model.

The successful diagnostic still uses a JAX mean kernel. It is NOT a completed
all-Pallas implementation. Next reproduce this mean in Pallas using measured
reduction-order candidates (contiguous/tiled pairwise trees) and compare full
mean bits, prefix and unchanged suffix controls. No input-dependent epsilon,
hardcoded row correction or tolerance relaxation is acceptable. Compile
rejections are separate from numerical failures and performance.

Original full Q remains inexact even with the exact input prefix, because
the separately compiled suffix differs from the unchanged monolithic oracle.
Production/default/BN/beam paths remain unchanged; no speed claim is made.
