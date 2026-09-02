# Strict Artgor exact-inference speed proof

Private Kaggle kernel
[`trydotatwo/tpu-artgor-exact-speed-proof`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-exact-speed-proof)
version 1 completed on eight TPU v5 lite devices.  The pinned source is
`3070839d4f04cff8fa58794024384c9bd98aa947`; runtime versions are JAX/jaxlib
0.10.2 and libtpu 0.0.42.1 with x64 enabled.

## Frozen gate

Each case uses 32,768 states per device, BF16 outputs, three warmups and 21
alternating synchronized A/B measurements.  The frozen requirement is zero
BF16 mismatches plus all three speed statistics at least 1.5x: ratio of
medians, every paired observation and a one-sided stratified bootstrap lower
99% bound.

| Corpus | Original JAX | Exact split | Median ratio | Minimum pair | Lower 99% | BF16 mismatches |
|---|---:|---:|---:|---:|---:|---:|
| legal seed 42 | 24.856 ms | 15.696 ms | 1.5836x | 1.5389x | 1.5779x | 0 |
| legal seed 142 | 24.807 ms | 15.610 ms | 1.5892x | 1.5695x | 1.5836x | 0 |
| legal seed 242 | 24.779 ms | 15.345 ms | 1.6148x | 1.5219x | 1.5859x | 0 |
| stress seed 43 | 24.739 ms | 15.406 ms | 1.6058x | 1.5921x | 1.6011x | 0 |
| stress seed 143 | 24.563 ms | 15.501 ms | 1.5846x | 1.5332x | 1.5755x | 0 |
| stress seed 243 | 24.818 ms | 15.434 ms | 1.6080x | 1.5352x | 1.5997x | 0 |

All 126 paired observations exceed 1.5x.  Across the six cases, the weakest
median ratio is **1.5836x**, weakest individual pair **1.5219x**, and weakest
bootstrap lower 99% bound **1.5755x**.

The result was independently recomputed from the downloaded raw pairs using
the frozen statistic implementation and original per-case bootstrap seeds;
all six recomputed gates pass.  The JSON SHA-256 is
`6147ffbca3a76ffa07c1356c34453c4cf4588b8538d7f5c6d2c0d526611097fe`.

## Scope

This proves the full model Q-inference component is at least 1.5x faster than
the unchanged JAX baseline under the frozen eight-TPU protocol.  It does not
claim 1.5x for a full beam depth or solver.  The matched production notebook
measures 1.1637x aggregate wall-speedup over four complete frame-runs; the
smaller parity benchmark previously measured 1.103x for complete beam depths.
