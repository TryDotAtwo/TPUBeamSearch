# Eight-device execution-boundary A/B v1

## Result

No candidate passed the frozen exact-Q gate.  The run did, however, remove the
main ambiguity left by the first eight-device experiment: neither the number of
devices nor the Python/JAX launch API causes the rare drift.  The Pallas banked
encoding is semantically exact; changing its surrounding compiled graph changes
the arithmetic schedule of later JAX Dense/LN work.

At local batch 16,384 per device, the fastest exact JAX control was typed
`shard_map`: 11.8814 ms on legal scrambles and 11.9682 ms on categorical
stress, or 11.032 and 10.952 million global states/s.  The current monolithic
Pallas-bank candidate took 7.1812/7.3120 ms (18.252/17.926 Mstate/s), nominally
1.654/1.637x faster, but retained 12/55 differing BF16 Q values.  It is
therefore rejected.  No 32K confirmation ran.

## Reproducibility

- Private Kaggle kernel: `trydotatwo/tpu-inference-execution-ab`, version 1.
- Public source: `88d6e42c4100578aa9478d3faf6b4f5d30adc01f`.
- Python 3.12.13, JAX/jaxlib 0.10.2, libtpu 0.0.42.1, NumPy 2.5.0.
- Eight visible and active `TPU v5 lite` devices in one process.
- Checkpoint/model-source/puzzle/legal-input/stress-input hashes exactly match
  the preceding exact eight-device v1 run; all hashes are retained in JSON.
- Five warmups and twelve paired forward/reverse synchronized samples.  Compile,
  first execution, host conversion and one-time bank construction are excluded.

This is full 150x24 embedding -> Dense 3600x1024 -> ten-block 1024-wide
LayerNorm ResMLP -> 30-score inference only.  No beam-search operation is
included.

## Eight-device matrix

| Variant | Legal ms | Stress ms | Legal Q mismatches | Stress Q mismatches |
|---|---:|---:|---:|---:|
| original `shard_map` | 11.9999 | 12.0581 | 0 | 0 |
| typed-BF16 `shard_map` | **11.8814** | **11.9682** | 0 | 0 |
| Pallas bank, no barrier | 7.1812 | 7.3120 | 12 | 55 |
| Pallas bank, pre-input-Dense barrier | 7.1571 | 7.4103 | 12 | 55 |
| Pallas bank, post-input-Dense barrier | 7.1656 | 7.2933 | 629,321 | 1,894,780 |
| Pallas bank, two-dispatch split | 7.5058 | 7.5803 | 93 | 427 |
| JAX encoding, two-dispatch split | 12.2069 | 12.3408 | 93 | 427 |
| Pallas `pmap` | 7.2041 | 7.2889 | 12 | 55 |
| Pallas, eight independent executables | 7.9108 | 7.8144 | 12 | 55 |

Original direct sharded `jit`, `pmap` and eight independent executables are all
elementwise identical to the oracle.  Their medians are within 5.5% of the
original `shard_map`; independent host orchestration is slower.  A direct
multi-device Pallas `jit` is not legal in this runtime because Mosaic custom
calls cannot be automatically partitioned; `shard_map` or `pmap` is required.

The important output-hash controls are stronger than mismatch counts:

- original `shard_map`, typed `shard_map`, original direct `jit`, original
  `pmap` and original independent executables share the oracle hash;
- Pallas `shard_map`, its pre-barrier arm, Pallas `pmap` and Pallas independent
  executables share one rejected hash on each corpus;
- JAX split and Pallas split share another rejected hash on each corpus.

Thus Pallas produces the same encoded BF16 tensor as JAX at the explicit split.
The split tail itself changes the downstream result.  Launch API and device
count do not repair or create the arithmetic difference.

## Witness replay and localization

The exact owner shards for legal rows 29,807 and 50,224 and stress row 29,369
were replayed on one TPU with the same 16K local shape.  The monolithic Pallas
candidate reproduces 6, 6 and 30 Q mismatches respectively.  The old one-device
prefix looked exact only because it did not contain these states.

For every witness, a diagnostic graph returning the encoded input, input Dense,
input LN/ReLU, every residual-block output and final Q makes Pallas and typed
JAX elementwise identical at every observed node.  This is intentionally not
treated as a contradiction: exposing intermediates changes fusion/scheduling.
Together with the unobserved replay, it proves the source formulas and embedding
values are not the defect and that the compiler context is causally relevant.

The previous compiled-HLO comparison found the only changed MXU convolution
schedule among 22 Dense operations at convolution index 20: the second Dense of
residual block 9 uses iteration bounds 2x16x1 in typed JAX and 1x22x1 in every
fast-encoding monolith.  The present result rules out input barriers and whole
encoding splits.  The next bounded experiment therefore places barriers and
real device-resident execution cuts directly around residual block 9 and its
second Dense, while preserving the exact full-Q gate.

## Other observations

The post-input-Dense barrier is a useful negative control: it changes hundreds
of thousands of Q values and materially changes argmin (97.12% legal, 90.54%
stress), despite nearly identical latency.  An optimization barrier is not a
numerical-preservation mechanism for this BF16 graph.

Compiler-reported static temporary storage is 758,780 kB for typed JAX and
129,892 kB for the Pallas monolith.  These are allocation estimates, not
measured HBM traffic, cache residency or MXU utilization.

Raw JSON, both logs and all StableHLO/compiled HLO are retained beside this
report.  No BN path or production default changed.
