# Artgor all-Pallas diagnostic v9

Eight Kaggle TPU v3 cores evaluated the production-only all-Pallas engine against
the unchanged `jax_model.apply` oracle and the published exact hybrid.

## Result

- The candidate was rejected on all six 16K-state/device corpora.
- Full BF16 Q mismatch counts were 2,887,465--2,891,812 on legal inputs and
  3,454,358--3,454,885 on stress inputs.
- On legal seed 42, median latency was 33.848 ms for all-Pallas, 12.372 ms for
  original JAX, and 7.764 ms for exact hybrid.
- Hybrid/all-Pallas median speed ratio was 0.2294; the paired bootstrap 99%
  lower bound was 0.2246. Thus all-Pallas was 4.36x slower than hybrid.
- Counting padded MXU work (49,807,360 FLOP/state), all-Pallas sustained about
  192.9 aggregate TFLOP/s, 12.24% of the nominal 8 x 197 TFLOP/s BF16 peak.

The earlier 44-output trace is not a valid exactness oracle because observing all
boundaries changes JAX lowering. The next experiment uses isolated Pallas
input/residual/head operators and feeds both reference and candidate boundaries
through the exact same JAX suffix executable. A zero-replacement control records
boundary-only drift relative to monolithic JAX.
