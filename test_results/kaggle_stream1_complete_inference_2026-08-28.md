# TPU Stream1 complete residual inference

Kaggle kernel `trydotatwo/tpu-stream1-complete-inference`, version 4,
completed on commit `13143f38db0b6cfc5dc0fe33fa47570ad5defbaa`.

## Exact checkpoint contract

- Logical state: `[B,120]`; aligned storage: `[B,128]`; classes: 120.
- Virtual one-hot input: `14400 -> 1536`, folded BN, ReLU.
- Hidden: `1536 -> 512`, folded BN, ReLU.
- Two residual blocks, each `512 -> 512 -> 512`, with both BNs folded.
- Output: `512 -> 24`; returned shape: `[B,24]`.
- Checkpoint weights are discovered from their shapes, not from a model name.

## JIT contract

Static per executable: architecture, batch shape, dtypes, and Pallas tiles.
Dynamic without recompilation when shapes remain unchanged: states, folded
weights, and folded biases.

The physical input may be padded, but virtual one-hot addressing consumes only
the first `STATE_LEN` values. Matrix padding is internal to each Pallas kernel
and is trimmed before the public result is returned.

## Real TPU result

- Backend: Kaggle TPU v5 lite, eight visible devices; JAX 0.10.2.
- Batch: 256 reachable Megaminx states.
- Compile plus first execution: 0.401090 s.
- Steady median over 31 samples: 0.463090 ms.
- Maximum absolute difference from the BF16 JAX reference: 0.125.
- Mean absolute difference: 0.0104523.
- Per-state argmax agreement: 95.3125% (244/256).

The CPU Pallas interpreter matches the hand-derived complete residual fixture
exactly. Real TPU and XLA reference reductions have different accumulation
orders, so bitwise BF16 equality is not expected. The production validation
gate requires finite output and maximum absolute error no greater than 0.25;
argmax agreement is retained as a diagnostic rather than hidden.

Kaggle required upgrading `libtpu` before importing JAX; the bundled runtime
was too old for the installed Pallas lowering.
