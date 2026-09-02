# All-Pallas exact diagnostic v3

Private Kaggle kernel
[`trydotatwo/tpu-artgor-all-pallas-exact-diagnostic`](https://www.kaggle.com/code/trydotatwo/tpu-artgor-all-pallas-exact-diagnostic)
version 3 completed on eight TPU v5 lite devices at source
`7888c0e548d111f53861d766193f41bee58df81a`.  This was the first run to pass
all Pallas/Mosaic compilation gates and reach operator-level arithmetic.

## Boundary result

The transparent model exposes 44 operator boundaries on three legal and three
categorical-stress seeds.  Both candidates have bitwise-exact prepacked
embedding outputs.

- `BK=128`, `BN=256`: the input Dense output is bitwise exact on all six
  corpora.  The first mismatch is consistently `input.layernorm_relu`.
- `BK=1024`, `BN=256`: the first mismatch is already `input.dense` on every
  corpus.  Legal seed 42 has 120 differing elements and stress seed 43 has
  174, establishing a reduction-schedule mismatch.  This arm is retained only
  as a negative control.

For `BK=128`, legal seed 42 has 163,288 differing LayerNorm+ReLU elements out
of 2,097,152 with maximum absolute error approximately 0.03.  Stress seed 43
has 154,177 differences with maximum absolute error approximately 0.02.
Because no candidate passes every boundary, the frozen protocol rejects both
before performance measurement.  No fusion or speed claim follows.

## Decision

`BK=128` is the exact Dense contract and the only candidate carried forward.
The next experiment is a causal LayerNorm arithmetic bundle with matched
monolithic, decomposed and materially split JAX controls, plus observable
Pallas checkpoints for mean, centered values, variance, inverse standard
deviation, affine output and ReLU.  Fusion remains blocked until the complete
44-boundary model is BF16 exact.

The safe JSON SHA-256 is
`d1da8da0abbbcabc11ce780e576fa1649753e5f3d68c7bc63ed22d14d6a3be34`.
Raw private logs are retained locally and are not published.
