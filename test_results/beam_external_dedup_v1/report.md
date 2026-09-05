# External dedup V1: benchmark shape error before TPU lowering

Kaggle ERROR. Source `163c88cbf83d8adce89192b5e01db20a66c0ee0a`;
eight TPU v5 lite devices; JAX/jaxlib 0.10.2, libtpu 0.0.42.1.
Partial JSON contains runtime inventory and zero completed cases.

The first N256 trace failed at the primitive's input validation:
`ValueError: count and threshold must be uint32 [1]`.
Global controls have shape [8,1,1]; shard_map retains all axes and each rank
gets [1,1,1]. The benchmark used c[0]/t[0], leaving [1,1]. The primitive
requires [1]. Changing the adapter to c[0,0]/t[0,0] fixes this shape mismatch.
No production primitive changed. There is no Mosaic lowering failure,
physical correctness result, HLO or execution timing from this run.

Regression: jax.eval_shape on the actual per-rank argument shapes reproduced
the same failure, then passed after the adapter fix. This is a trace-level
test, not TPU evidence. V2 must establish physical compilation/correctness.
