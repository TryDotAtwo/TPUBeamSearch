# TPUBeamSearch experiment ledger

This file separates measured facts from hypotheses so benchmark conclusions do
not silently turn into architecture assumptions.

## Fixed model contracts

- Artgor checkpoint: `q555_2k_BEST.pt`, 24,757,807 parameters.
- State shape: 150 categorical `uint8` values.
- Embedding shape: `150 x 24`; therefore `NUM_CLASSES=150` and `EMBED_DIM=24`.
- Input dense: `3600 -> 1024`, LayerNorm, ReLU.
- Trunk: 10 residual blocks, each containing two `1024 -> 1024` dense layers
  and two LayerNorm operations.
- Head: `1024 -> MOVE_COUNT`, with `MOVE_COUNT=30` for this checkpoint.
- Inference storage/compute contract under comparison: BF16 model tensors.

## Measured facts

- Original JAX full model at local batch 16,384: approximately 1.39M states/s.
- Standalone Pallas LayerNorm BM128/BM256 is about half the throughput of JAX
  LayerNorm; BM512 and BM1024 exceed 16 MiB VMEM in the tested layout.
- For the embedding model, ordinary embedding gather beat virtual one-hot MXU.
- Per-layer Dense+LayerNorm fusion improved an isolated layer at BM128, but the
  best absolute isolated result was BM256/BK256/BN512.
- Incremental residual-block diagnostic on the same hidden input:
  - Dense1: Pallas/JAX 0.606x; 99.99936% BF16 outputs exact.
  - Dense1+LN+ReLU: 0.394x; 90.566% exact.
  - Two-kernel residual block: 0.283x; 72.146% exact.
- The old `per_layer` implementation was not block fusion: it launched two
  Pallas kernels and materialized the activation between them.

## Corrected mistakes

- `150 x 24` was temporarily misread as 24 categories. Correct meaning is 150
  categories with embedding width 24.
- A repeated `0..149` state is valid but degenerate, not out of domain.
- Full JAX versus fragmented Pallas was not a fair measurement of Pallas's
  block-level potential. It remains useful only as an end-to-end regression.
- Finite output alone is not correctness. Full-model acceptance also requires
  numerical bounds and at least 99% argmax agreement on diverse valid states.

## Active hypotheses

1. Reusing `hidden1`, dense output, and accumulator scratch inside one kernel
   should reduce the large two-kernel residual-block penalty.
2. Current Dense tiling leaves throughput on the table even before LayerNorm;
   BM/BK/BN must be swept rather than inherited from the isolated fusion run.
3. FP32 statistics may improve full-model ranking stability but can reduce
   throughput and differ from the checkpoint's original BF16 execution.
4. A block candidate that wins at batch 4096 may not remain Pareto-optimal at
   16,384 or 32,768, so promotion must remeasure rather than extrapolate.
5. Small per-block BF16 differences may accumulate through ten blocks; block
   correctness does not authorize full-model scaling without the output-head
   argmax gate.

## Comprehensive staged run

1. Screen 32 residual-block candidates at batch 4096:
   BM 128/256, BK 128/256, BN 256/512, BF16/FP32 statistics, and two-kernel
   versus one-kernel boundaries.
2. Promote only the three fastest correctness-valid candidates and remeasure
   them at batches 16,384 and 32,768.
3. Use the winning tiling for full-model `separate`, `per_layer`, and
   `per_block` comparisons against original JAX at batch 16,384.
4. Run 1/8-TPU and 128-chunk scan scaling only after a Pallas full-model
   candidate passes the numerical and argmax gates.
5. Atomically checkpoint JSON after every candidate; compile/VMEM failures are
   recorded and do not terminate the remaining sweep.
