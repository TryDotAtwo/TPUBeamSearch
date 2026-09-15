# Response stage isolation V2

Source: `5cba03c052f07ae35d75aa57b74304d3fb5e4197`; launcher `0220108`.
Artifacts: `test_results/response_stage_isolation_v2/` (including full Kaggle log).

All seven nested reports match the coordinator report. Each reports JAX/jaxlib
0.10.2, libtpu 0.0.42.1 and eight distinct TPU v5 lite devices (IDs 0..7).
Every stage has lowered MLIR and a process log; all five successful stages
also have compiled HLO. Parent return code is 1, not an incomplete bundle.

| Stage | Return code | Compiled |
|---|---:|---|
| packing | -6 | no |
| packing_control | 0 | yes |
| packing_selection | 0 | yes |
| exchange | 0 | yes |
| receive | 0 | yes |
| planes_to_wire | 0 | yes |
| composition | -6 | no |

The receive unsigned-min replacement and rectangular byte store now pass
physical TPU compilation, unlike V1. This does not validate their execution.
Both packing and composition still abort in VectorLayout::join with the
layout-rank check `(1 vs. 2)`.

The control and selection prefixes compile. Thus the isolated interval/error,
offset and peer-selection expressions are accepted in those diagnostic
contexts. This does not prove that their layouts remain compatible inside the
complete packing kernel. The next experiment must incrementally retain the
conditional region, first DMA, second DMA and gather/output operations, with
observable outputs preventing elimination. No precise failing source expression
has yet been established; the compiler stack alone is insufficient.

This run is compile-only: no correctness, throughput, latency, overlap or
end-to-end beam speedup is measured. The full architecture goal remains open.
