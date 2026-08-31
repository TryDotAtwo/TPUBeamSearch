# Comprehensive Pallas LayerNorm sweep

> Audit note (2026-08-31): historical timings and metrics below are unchanged.
> The saved argmax metric is not the original Q-beam's minimizing action
> selector; equal aggregate errors do not prove pairwise tensor equality.
> The cause of numerical divergence is unresolved. See the
> [source/code audit](../docs/research/2026-08-31-tpu-coding-research.md).

Private Kaggle TPU v3-8 kernel
`trydotatwo/tpu-layernorm-comprehensive-sweep`, version 1, evaluated 32
residual-block configurations and promoted the three fastest correctness-valid
candidates before running the complete 10-block checkpoint.

## Screening: batch 4,096

- 24 candidates passed the block numerical gate.
- 8 candidates were rejected by scoped VMEM, all of them one-kernel
  `per_block` configurations with `BM=256`.
- The rejected layouts requested 16.06-16.36 MiB against the 16.00 MiB limit.
- Fastest screening candidate:
  `per_block, BM128, BK256, BN512, FP32 statistics` at 5.295M states/s.
- The matching fastest two-kernel candidate was within measurement noise:
  `per_layer, BM256, BK256, BN512, FP32 statistics` at 5.295M states/s.

Group summary:

| Boundary / statistics | Valid | Best states/s | Mean states/s |
|---|---:|---:|---:|
| one-kernel block / BF16 | 4 | 4.528M | 4.056M |
| one-kernel block / FP32 | 4 | 5.295M | 4.816M |
| two-kernel block / BF16 | 8 | 4.522M | 4.087M |
| two-kernel block / FP32 | 8 | 5.295M | 4.743M |

FP32 LayerNorm statistics were consistently faster in this Pallas layout and
also reduced block-level maximum error from 0.1875 to 0.125. Bit-exact
fraction is not a sufficient quality metric here: FP32 statistics had lower
numerical error despite fewer exact BF16 elements.

## Promotion

| Candidate | Batch 16,384 | Batch 32,768 | max abs | mean abs |
|---|---:|---:|---:|---:|
| one-kernel BM128/BK256/BN512 FP32 | 7.230M | 7.481M | 0.125 | ~0.00129 |
| two-kernel BM256/BK256/BN512 FP32 | **7.466M** | **8.030M** | 0.125 | ~0.00129 |
| two-kernel BM128/BK256/BN512 FP32 | 6.800M | 7.041M | 0.125 | ~0.00129 |

The one-kernel boundary wins the small-batch screen but loses to two kernels
with BM256 by 3.2% at batch 16,384 and 6.8% at batch 32,768. BM256 one-kernel
cannot be tested with the current scratch layout because it exceeds VMEM.

## Full checkpoint: batch 16,384

The promoted BM128/BK256/BN512 FP32 tile was used for all Pallas fusion
boundaries so the comparison changes only the boundary.

| Implementation | states/s | vs JAX | max abs | mean abs | argmax |
|---|---:|---:|---:|---:|---:|
| original JAX/XLA | **1.386M** | 1.000x | - | - | reference |
| Pallas separate | 0.577M | 0.416x | 1.125 | 0.141 | 73.69% |
| Pallas per-layer | 0.547M | 0.395x | 1.125 | 0.141 | 73.69% |
| Pallas per-block | 0.565M | 0.408x | 1.125 | 0.141 | 73.69% |

All Pallas outputs were finite, but none passed the required 99% argmax gate.
The identical full-model numerical result across all three boundaries confirms
that one-kernel fusion preserves the intended Pallas arithmetic; accumulated
Pallas-versus-XLA arithmetic differences, not the kernel boundary, cause the
ranking divergence. Scaling was correctly skipped.

## Consequences

1. One-kernel block fusion is viable only at BM128 with the current VMEM
   layout and is not yet faster at production-sized batches.
2. The next correctness experiment must record error after every residual
   block and head agreement after replacing the first 1..10 blocks with
   Pallas. This will locate how ranking agreement decays through depth.
3. Dense and LayerNorm arithmetic variants must be evaluated on that staged
   curve before further end-to-end performance work.
4. 1/8-TPU scaling remains blocked by full-model correctness, not by lack of a
   fast block microbenchmark.
