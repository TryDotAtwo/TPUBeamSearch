# Incremental LayerNorm residual-block diagnostic

Private Kaggle TPU v3-8 kernel `trydotatwo/tpu-layernorm-block-diagnostic`
compared JAX and Pallas on exactly the same 16,384 hidden vectors produced by
the original JAX input prefix from deterministic valid states.

| Level | JAX states/s | Pallas states/s | Pallas/JAX | max abs | mean abs | exact |
|---|---:|---:|---:|---:|---:|---:|
| Dense1 | 37.416M | 22.662M | 0.606x | 0.5 | 0.00000164 | 99.99936% |
| Dense1 + LN + ReLU | 27.685M | 10.902M | 0.394x | 0.125 | 0.00020472 | 90.566% |
| Full residual block | 20.369M | 5.758M | 0.283x | 0.1875 | 0.00054607 | 72.146% |

The first Dense is already 1.65x slower in the current hand-written Pallas
tiling, although almost every BF16 output is bit-identical. LayerNorm then adds
both a second performance loss and the expected reduction-order differences.
The previous `per_layer` block is two independently launched Pallas kernels,
so it materializes the first normalized activation between the two dense
layers and reaches only 28.3% of JAX block throughput.

The next valid experiment is therefore one Pallas kernel for the complete
`Dense1 -> LN1 -> ReLU -> Dense2 -> LN2 -> skip -> ReLU` block, reusing VMEM
scratch between the two MXU pipelines. It must be compared against both JAX
and the measured two-kernel Pallas block before any full-model conclusion.
