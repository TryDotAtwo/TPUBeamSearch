# Shapes, layout and pipelines

Checked: 2026-08-31. Use for Pallas block geometry, normalization padding,
reductions, DMA or VMEM allocation questions.

## Three shape contracts

Keep logical model dimensions, allocated storage dimensions and kernel tiles
distinct. For a logical width 130 stored at width 256, LayerNorm statistics
use only the 130 valid values: masked sum divided by 130, then masked squared
deviations divided by 130. A plain mean over 256 changes the operator even
when the tail is zero. This describes centered variance; numerical matching
also preserves the reference estimator, reduction order and rounding.
Centered variance versus `E[x²] - E[x]²` is a separate A/B choice. Preserve
epsilon, statistic dtype, affine parameters and residual/activation order.
Zero invalid output lanes again if the affine bias makes them nonzero.

For Dense reduction padding, initialize finite neutral tails (normally zero
inputs and weights). Do not depend on multiplication by zero to sanitize
uninitialized NaNs. Padded output lanes are not additional actions: slice or
mask them, using positive infinity for minimizing selection or negative
infinity for maximizing selection. Preserve logical move count and validity.
See [numerical validation](numerical-validation.md).

[TPU kernel details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
qualify the common trailing `(8, 128)` layout: for blocks of rank at least two,
the last two block dimensions must be divisible by 8 and 128 respectively,
**or equal the corresponding whole-array dimension**. Thus a whole width
130 is not automatically illegal. Legal block shape does not guarantee every
operation or layout compiles efficiently; rank-one cases and dtype/layout
constraints need their own check. Transposes and reductions on trailing axes
may be expensive. CUDA warp/occupancy rules do not describe this layout.

## Lifetime, not cache intuition

VMEM/SMEM are software-managed on-chip SRAM, not CUDA-style transparent
caches. Budget input/output windows, accumulators, scratch, pipeline buffers,
and overlapping lifetimes. `pl.ANY` means unconstrained placement, not
guaranteed HBM. Default pipelines already stage transfers; explicit double
buffering or lookahead is an A/B candidate, not an automatic improvement.
Wait for asynchronous transfers before consuming destinations or reusing
sources/buffers/semaphores. See [TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).

Ordinary TPU `pallas_call` grid traversal is sequential lexicographic execution,
except explicitly parallel multicore dimensions. Consecutive input-window
reuse can avoid transfers. Updates to one output window must stay consecutive:
place a reduction dimension so partial sums are initialized once, accumulated
across that window, then finalized before advancing. See
[grid semantics](https://docs.jax.dev/en/latest/pallas/tpu/details.html) and
[matmul tiling](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html).
BM/BK/BN tiles need not equal one hardware MXU array.

A compiler's scoped-VMEM allocation limit is not a measurement of total
physical VMEM. Save the rejection and allocation context; compare runtime
parameter arguments with captured weights when caller lowering differs.
Historical limits and negative buffering experiments are in
[case studies](case-studies.md), not universal tuning constants.
