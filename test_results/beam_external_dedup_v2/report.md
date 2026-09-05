# External dedup V2: count-buffer block geometry rejected

Kaggle ERROR. Source `503043b0375a4b78bb761da41156823d125d9a90`.
Runtime inventory: eight TPU v5 lite, JAX/jaxlib 0.10.2, libtpu 0.0.42.1.
No completed cases, executable HLO, correctness results or timings.

The corrected shard adapter passes input validation. TPU lowering rejects
`beam_external_neutral_counts` output 1: block (1,128) over array (2,128).
The row block neither spans the whole row dimension nor is divisible by 8.
This is a diagnosed BlockSpec geometry error, not a native compiler abort.

Fix: arrange per-tile counts as aligned column segments in [1,tiles*128]
instead of [tiles,128]. Each program still writes one [1,128] segment, with
index map (0,tile). Count reduction and external [1,128] result are unchanged.
Allocated element count is unchanged. No source Stream3 semantics change.

A local regression traversing the called output block specifications rejected
the exact (1,128)/(2,128) geometry before the fix. This structural test checks
the documented TPU shape rule, not physical compiler correctness. Value-level
interpreter and original C++ oracle checks remain required; V3 is the physical gate.
