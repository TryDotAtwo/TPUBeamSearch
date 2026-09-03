# Stream2 hash/goal parity implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. No subagents.

**Goal:** Port the immediate-child hash and exact-goal stage, with independent
source-derived checks, without materializing child states in HBM.

**Architecture:** One candidate lane means one parent/move pair. Four uint32
XOR accumulators preserve Hash128. Full-storage goal comparisons preserve padding
semantics. Invalid tail lanes have explicit validity and zero outputs.

**Tech Stack:** Python, NumPy, JAX/Pallas, pytest; source C++ oracle when available.

**Spec:** `docs/TPU_ARCHITECTURE.md` and read-only
`D:/100XH100/cuda/stream2.cu`, `src/hash.cpp`, `src/state.cpp`.

## Constraints

One TPU session; no BN/default changes; no subagents or network reconfiguration.
K1/K2 and bounded solved-record collection are separate unimplemented gates;
this primitive must not silently advertise those features.

### Task 1: Immediate hashes and exact goal flags

Files: `src/tpu_beam_search/beam_stream2.py`, `tests/test_beam_stream2.py`.
Interface: `pallas_hash_goal(parents, generators, central, zobrist_words, count,
tile_candidates=128, interpret=False)` -> `(hash_words[4,N], goal[1,N], valid[1,N])`.
Input parents uint8[B,S], generators int32[M,S], central uint8[S], Zobrist uint32
[4,S*C], count uint32[1]. Output capacity rounds B*M to tile_candidates.

- [x] Test candidate order parent*M+move, noncommuting permutations, high hash
  bits, exact goal versus hash collisions, zero valid parents and partial tail.
  Compute expected states using `parents[:, generators]` only in test oracle,
  then XOR every position's independently prepared Zobrist words.
- [x] Run `python -m pytest tests/test_beam_stream2.py -q`, observe missing module.
- [x] Implement only Pallas gather/XOR/goal operations inside pallas_call.
  Loop positions with lax.fori_loop, mask invalid lanes and never emit child states.
- [x] Run interpreter and complete affected test set; record real TPU compilation
  as a separate outstanding gate.

### Source parity follow-up

Build a local executable against the selected original C++ hash/state source,
record source SHA256, and compare identical fixture bytes. This verifies shared
host/device helpers only, not CUDA execution. Actual GPU replay, K1/K2, owner/
shard arithmetic, dedup, final history and 8-TPU replay remain mandatory for the
whole-architecture acceptance; no completion claim at this subproject boundary.

### Added independently tested primitives

`beam_hash.py`: port unsigned64 mixing as uint32 pairs; tests first in
`tests/test_beam_hash.py` (zero, all ones, high words, modulo 1/3/7/8/256/1024/
UINT32_MAX). Compare original C++ owner and shard outputs on the same Stream2
fixture. Five routing/source tests passed after missing-module red.

`beam_dedup.py`: diagnostic capacity 128..4096 bitonic sort, not a scalable
HBM sort. Tests first in `tests/test_beam_dedup.py`: duplicate Hash128, opposite
payload/parent tie-break, high parent words, route ties, empty/full/partial valid
counts at UINT32_MAX threshold. Six tests passed after missing-module red.
The adapter additionally links original stream4.cpp for an independent differential
case. Full-port missing items are tracked in `docs/TPU_PORT_LEDGER.md`.
