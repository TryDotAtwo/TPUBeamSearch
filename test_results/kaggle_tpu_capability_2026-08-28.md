# Kaggle TPU beam-search capability smoke — 2026-08-28

Kernel: `trydotatwo/tpu-beamsearch-capability-smoke`, private, version 3, COMPLETE.

Environment: Python 3.12.13, JAX 0.10.2, TPU backend, 8 local devices arranged as coordinates `(x=0..1, y=0..3)`.

Important environment finding: Kaggle's bundled JAX and TPU runtime were incompatible for Pallas. `pip install --upgrade libtpu` before importing JAX fixed Pallas execution.

Verified results:

- JAX `uint64` hash arithmetic: correct.
- Pallas uint32 vector kernel: correct.
- Pallas BF16 matrix multiply with FP32 accumulator: correct for `256x512 @ 512x512`.
- Eight-core `psum` of the exact 307201-bin uint32 selection histogram: correct; 1,228,804 bytes/core; steady 1.642 ms.
- Lexicographic sort of 262144 `(uint32_hi, uint32_lo)` hash keys: correct; steady 0.968 ms after 12.18 s compilation.
- Attached Megaminx MLP weights inspected: input `14400 -> 2048`, hidden `2048 -> 512`, eight residual blocks with two `512 -> 512` linears each, scalar output.

Interpretation:

- The GPU algorithm's global histogram threshold maps directly to an eight-core TPU collective; a global top-k is unnecessary.
- Pallas is usable on Kaggle, but the notebook must refresh `libtpu` first.
- MXU operations must accumulate BF16 products into FP32.
- The unresolved work is performance/correctness at production beam sizes: full MLP inference, Hash128 sort/reduce/compact, ownership exchange, and materialization.

Raw evidence:

- `test_results/kaggle_tpu_capability_v3/tpu_capability_results.json`
- `test_results/kaggle_tpu_capability_v3/tpu-beamsearch-capability-smoke.log`
