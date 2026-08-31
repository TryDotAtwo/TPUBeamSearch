# Per-depth LayerNorm arithmetic diagnostic

> Audit note (2026-08-31): the measurements below are retained, but the causal
> interpretation is not established. The segmented-JAX oracle and fused-JAX
> suffix lack a JAX-only boundary control. `argmax` is also not this Q-beam's
> minimizing action selector, and equal summaries do not prove identical
> tensors. See the [audit and required controls](../docs/research/2026-08-31-tpu-coding-research.md).

Private Kaggle TPU v3-8 kernel `trydotatwo/tpu-layernorm-depth-diagnostic`,
version 1, measured four arithmetic/layout configurations on 8,192 valid
states. Every depth used the same original JAX input prefix.

For each residual depth the run recorded:

- isolated Pallas block error on the correct JAX hidden input;
- cumulative hidden drift after replacing the first 1..N blocks with Pallas;
- final output after executing the remaining blocks and head with JAX.

## Boundary result

`per_block BM128` and `per_layer BM256` produced identical numerical results
at every depth for the same statistics mode. Kernel boundary and tile size are
therefore performance choices here; neither causes the correctness failure.

## BF16 statistics curve

| Depth | isolated mean abs | cumulative mean abs | cumulative cosine | hybrid output mean abs | argmax |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000545 | 0.000545 | 0.999991 | 0.17224 | 73.01% |
| 2 | 0.000440 | 0.001888 | 0.999940 | 0.17144 | 72.13% |
| 3 | 0.000560 | 0.005510 | 0.999676 | 0.17210 | 71.92% |
| 4 | 0.000411 | 0.006458 | 0.999572 | 0.17182 | 72.31% |
| 5 | 0.000364 | 0.007238 | 0.999418 | 0.17077 | 73.00% |
| 6 | 0.000331 | 0.008145 | 0.999292 | 0.17218 | 72.68% |
| 7 | 0.000247 | 0.007912 | 0.999230 | 0.17158 | 72.72% |
| 8 | 0.000175 | 0.007305 | 0.999242 | 0.17217 | 72.24% |
| 9 | 0.000517 | 0.007438 | 0.999027 | 0.17229 | 72.11% |
| 10 | 0.000619 | 0.005869 | 0.999384 | 0.17228 | 72.45% |

## FP32 statistics curve

| Depth | isolated mean abs | cumulative mean abs | cumulative cosine | hybrid output mean abs | argmax |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.001295 | 0.001295 | 0.999973 | 0.17494 | 72.14% |
| 2 | 0.001497 | 0.002538 | 0.999922 | 0.17585 | 71.77% |
| 3 | 0.002842 | 0.005174 | 0.999742 | 0.15878 | 72.84% |
| 4 | 0.001838 | 0.005869 | 0.999666 | 0.15434 | 72.85% |
| 5 | 0.001761 | 0.006506 | 0.999548 | 0.15085 | 72.47% |
| 6 | 0.001520 | 0.007329 | 0.999445 | 0.15095 | 73.24% |
| 7 | 0.001334 | 0.007125 | 0.999391 | 0.14969 | 73.11% |
| 8 | 0.000491 | 0.006595 | 0.999393 | 0.14837 | 73.12% |
| 9 | 0.001468 | 0.006667 | 0.999228 | 0.14632 | 73.60% |
| 10 | 0.001339 | 0.005174 | 0.999515 | 0.14048 | 73.93% |

## Interpretation

There is no late-depth cliff and no uniquely catastrophic residual block.
Block 3 has the largest isolated deviation in both modes, but ranking agreement
has already fallen to 72-73% after only block 1.

The decisive observation is amplification: BF16 block 1 differs from JAX by
only 0.000545 mean absolute hidden value with cosine 0.999991, yet the unchanged
JAX suffix expands that into 0.172 mean output error. The checkpoint is highly
sensitive to the exact arithmetic trajectory. Additional Pallas blocks change
hidden drift, but barely change the already-low ranking agreement.

FP32 LayerNorm statistics reduce the final output error as depth increases,
reaching 0.14048 at depth 10, but do not restore rankings. One-kernel fusion is
numerically faithful to two-kernel Pallas and cannot solve this arithmetic
sensitivity by itself.

## Next experiment

The next arithmetic A/B should replace only one operator at a time in block 1:

1. JAX Dense + JAX LayerNorm (control);
2. Pallas Dense + JAX LayerNorm;
3. JAX Dense + Pallas LayerNorm;
4. Pallas Dense + Pallas LayerNorm;
5. BF16 versus FP32 statistics for cases 3-4.

Each block-1 output must then pass through the identical JAX blocks 2-10 and
head. This directly attributes the ranking collapse to Dense reduction order,
LayerNorm reduction order, or their interaction before attempting more kernel
optimization.
