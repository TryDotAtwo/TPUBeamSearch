# Response stage isolation V1

Source `739ab3d980b5d19f2411ede9449b65b3d8052ec7`, launcher `0876c66`.
Artifacts: `test_results/response_stage_isolation_v1/`.
All five subprocesses ran. Each reports the expected source, JAX/jaxlib 0.10.2,
libtpu 0.0.42.1 and eight TPU v5 lite devices. This experiment compiles runtime
arguments only: no kernels execute and no correctness or latency is measured.

| Stage | Return code | Compile result |
|---|---:|---|
| packing | -6 | Native VectorLayout::join assertion, 1 vs. 2 |
| exchange | 0 | Compiled HLO saved |
| receive | 1 | Unsupported arith.minui legalization |
| planes_to_wire | 1 | Unsupported shape cast |
| composition | -6 | Native VectorLayout::join assertion, 1 vs. 2 |

The coordinator successfully preserved later diagnostics after the packing
abort. All stages saved precompile MLIR; only exchange saved compiled HLO.

## Source attribution

Receive compilation identifies `beam_final_receive.py:26`, the unsigned
`jnp.minimum(counts, uint32(128))` in summarize. Preserve raw overflow detection
and bounded signed summation when replacing this unsupported operation.

Planes-to-wire compilation identifies `beam_final_transport.py:39`, column
assignment of a uint8 vector; the rejected reshape is vector<128xi8> to
vector<128x1xi8>. A new layout must preserve little-endian bytes and padding.

Packing alone reproduces the composition's native layout assertion. This
narrows one failure to packing but does not identify its source expression;
further isolated operator variants are required. Repairing packing alone is
insufficient: receive and byte conversion are independent compilation blockers.

Exchange compile success is not remote-DMA execution acceptance. After repairs,
retain the unchanged 11-fixture/33-epoch correctness gate and eventually the
full GPU/8-TPU multi-depth replay. No speed or overlap claim follows from this run.
