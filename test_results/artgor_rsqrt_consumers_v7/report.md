# V7: rank-one BF16 output block rejected by TPU lowering

Source eadd7044467c92a8dcd6a3c7253a561cbb168a11, launcher 3ac45a0.
The run reached the first Pallas scalar FP32 consumer and failed to compile.
Its input was FP32 but its output BF16. A rank-one block of128 did not cover
the local16384 array and was smaller than the BF16 tiling size256. The precise
compiler rejection and partial results are preserved in JSON and both logs.

Only `consumer_scalar_fp32_jax` at16K completed. This is not a completed
JAX/Pallas A/B and establishes neither numerical equivalence nor performance.

Fix: rank-one blocks use256, or the whole array when smaller; grid count uses
the same block size. Matrix blocks remain128rows. A regression checks actual
traced Pallas output mappings against the reported TPU legality constraint,
including full-array128, array256 and array16384. Before the fix two cases
failed; after the fix all15 consumer tests pass. Interpreter tests had missed
the target-specific restriction. The mapping regression is not a TPU compile
proof: the next Kaggle version must validate actual lowering.

Full local regression after the fix:492passed in137.84s. No production model,
BN path, beam search or notebook default was changed.
