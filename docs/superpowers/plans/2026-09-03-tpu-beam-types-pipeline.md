# TPU beam types and first pipeline implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. No subagents.

**Goal:** Establish lossless TPU metadata types and an executable buffered Pallas
metadata packing kernel before integrating the complete beam stages.

**Architecture:** Flat uint32 SoA preserves the source record's bits. An explicit
nested Pallas pipeline packs separate fields into this transport representation;
buffer count is configurable from the first implementation.

**Tech Stack:** Python, NumPy, JAX/Pallas, pytest.

**Spec:** `docs/TPU_ARCHITECTURE.md`

## Global constraints

No subagents. Preserve BN/defaults and unrelated artifacts. One TPU session.
Karing untouched. Logical Hash128/parent64 are never narrowed. No performance
claim before real TPU measurements. This is the first subproject, not the whole
beam implementation; subsequent stage plans must retain all spec requirements.

### Task 1: Representation contract

Files: create `src/tpu_beam_search/beam_types.py`, `tests/test_beam_types.py`.
Interfaces: `BeamStorage(STATE_LEN, MOVE_COUNT, capacity, WORLD_SIZE=8)`;
`pack_candidates(hashes, parents, scores, routes, capacity=...)` returns uint32
`[8, padded_capacity]`; `unpack_candidates(words, count=...)` restores host arrays.

- [x] Add hand-checked high-word fixtures, malformed ranges, capacity and padding
  tests. Example hash `(1<<127)+7`, parent `(1<<63)+9` must produce words
  `[7,0,0,2147483648,9,2147483648,11,0x10002]`.
- [x] Run `python -m pytest tests/test_beam_types.py -q`; verify missing module.
- [x] Implement validated host preparation with NumPy, no host search hot path.
- [x] Run the tests; include 120/24 and 150/30 state/move configurations.

### Task 2: Buffered Pallas metadata packing

Files: create `src/tpu_beam_search/beam_transport.py`,
`tests/test_beam_transport.py`.
Interface: `pallas_pack_candidates(hash_words, parent_words, scores, routes,
tile_candidates=128, buffer_count=2, pipelined=True, interpret=False)`; uint32 inputs with shapes
`[4,N]`, `[2,N]`, `[1,N]`, `[1,N]`; output `[8,N]`.

- [x] Test all planes against explicit concatenation for multiple distinct tiles,
  including high uint32 bits and buffer_count=2/3; reject dtype/shape/tile errors.
- [x] Run `python -m pytest tests/test_beam_transport.py -q`; verify missing module.
- [x] Implement `pltpu.emit_pipeline` with `pl.Buffered` on every input/output
  BlockSpec inside an HBM outer pallas_call; no host concatenate in production.
- [x] Run interpreter tests, then TPU interpreter race detection for multiple
  tiles. Treat unsupported simulation as an explicit limitation, not a pass.
  Local JAX 0.10.1 requires a simulated abstract TPU geometry for nested pipeline
  DMA tiling; plain interpret=True on a CPU is insufficient. Use InterpretParams
  with race detection. Two output buffers, independently of 2/3 input buffers.
- [x] Run full pytest and `git diff --check`; record results in this plan.
  Full suite: 516 passed in 157.45 s. Subsequently added the serialized-control
  switch and reran the entire affected type/transport test set: 18 passed in
  6.97 s (two additional serialized cases). No physical TPU run yet.

### Integration boundary

Next independent subprojects: Stream2 hash/goal and exact field arithmetic;
fixed-capacity compact/sort/dedup; remote DMA epochs and shard collector;
Stream4 histogram/threshold; final requests/responses/history and full-depth
replay. Keep local pipeline and remote ring probes as standalone A/B controls.
Do not claim this packer implements remote exchange or the search scheduler.
