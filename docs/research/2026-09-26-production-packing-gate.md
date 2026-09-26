# Production final packing: eight-TPU execution gate

Private Kaggle kernel: `trydotatwo/tpu-beam-production-packing-execution`, V1.
Source SHA: `e71c92261811961307d3f2c13d93f36ad45d6a06`.
Launcher commit: `1f0f0f5`; it verifies the detached checkout SHA and a clean
worktree before executing. Submitted 2026-09-26; first observed status QUEUED.
Do not restart while QUEUED or RUNNING. Keep one TPU session.

The prior split-gather V2 result was one unsharded Pallas call. It inventoried
eight TPU devices but did not test all eight. The production gate uses a
`shard_map` over eight distinct TPU v5 lite devices. On each it runs the actual
`pallas_pack_final_chunk`, not the diagnostic `make_probe`. Its local input ABI
is `[32 or 35,2048]` uint32 HBM payload, `[3,128]` intervals, `[1]` chunk and
`[1,128]` prior error. The leading global dimension is the device axis.

For both 32-plane isolation and 35-plane response-caller shapes, one compiled
executable processes two fixture rounds. Mixed cases cover tile crossing,
second chunk, empty, malformed bounds, rank error, prior error, final HBM tile
and exhausted chunk. The all-live round sends 128 records for every device and
every peer. Payloads use deterministic full-range uint32 words plus explicit
high-bit sentinels. Acceptance requires exact wire and control arrays, shapes,
uint32 dtype, SHA-256 and per-device equality. Lowered MLIR and compiled HLO
are saved per plane shape. The JSON is saved before compilation/execution and
after each round to preserve partial evidence on failure.

No timing or RDMA is measured. Even a passing result will establish only
production packing correctness on eight devices; response composition, full
epochs, history and beam replay remain separate gates. The runner's output
will be downloaded to `test_results/production_packing_execution_v1/` after
terminal status, with full Kaggle log and a source/runtime/device audit.
