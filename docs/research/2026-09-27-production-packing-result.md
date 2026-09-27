# Production packing V1: eight-device correctness passed

Downloaded the terminal COMPLETE run on 2026-09-27 to
`test_results/production_packing_execution_v1/`, including JSON, full Kaggle
log, lowered MLIR and compiled HLO for both shapes.

Source matches `e71c92261811961307d3f2c13d93f36ad45d6a06`.
Runtime: JAX/jaxlib 0.10.2, libtpu 0.0.42.1; eight distinct TPU v5 lite
devices (IDs 0 through 7).

Production 32-plane and stress 35-plane packing each passed mixed and
all-live scenarios on all eight devices. Every result reports exact wire
and control, matching SHA-256, uint32 dtype, expected shapes and zero maximum
absolute error. Wire shape is [8,8,planes,128]; control is [8,8,2,128].
The mixed fixture covers crossing tiles, next chunk, empty, bounds error,
rank error, prior error, final HBM tile and exhausted chunk.

This establishes production packing execution correctness, not RDMA,
response composition, end-to-end beam correctness or speed. No timing
claim follows from this gate.

After terminal success, submitted the prepared private response gate as V5:
`trydotatwo/tpu-beam-response-epoch-gate`, source
`d5c8a45f02a9a71e05da2f5584c887fbf968bda3`, launcher `b4a0d13`.
Launcher syntax checked before submission. Next inspect its isolated
compile gates and all 11 fixtures / 33 epochs before advancing to the
prepared state-width bridge. Do not restart queued/running V5.
