# Public TPU expert follow-up

Date: 2026-08-31. This is a summary of expert recommendations, not new runtime
evidence. It follows the [code and documentation audit](2026-08-31-tpu-coding-research.md).

## Disclosure and context

The maintainer explicitly authorized sharing TPUBeamSearch materials with
project experts and keeping them in public GitHub. Repository visibility was
checked with `gh repo view`: `PUBLIC`.

One shared request went to `multigpu_beam` and `multigpu_mlp`, including the
public audit at commit `f166ad1`, checkpoint architecture, historical full-model
timings, missing JAX-only control, score-direction mismatch, Dense rounding
witness, synthetic input limitations and parameter-capture findings. No
credentials or unrelated private data were included. Both experts answered.

They are peers with beam-search/MLP context, not independent TPU hardware
measurements. Agreement with our audit is not proof of a numerical root cause.

## Recommendations retained

1. Check the reference first: original whole-model JAX, separately compiled
   JAX blocks and the exact JAX suffix used by each hybrid. Measure a candidate
   against the same-suffix reference before interpreting error amplification.
2. Cross Dense and LayerNorm implementations independently: JAX/JAX,
   Pallas/JAX, JAX/Pallas and Pallas/Pallas. Separate the first and second
   sublayers when attributing an effect.
3. Treat BF16 dot-result/bias rounding, normalization statistic precision,
   epsilon and logical width as distinct arithmetic choices. The CPU witness
   does not quantify any of their contributions on TPU.
4. Use the consumer's minimizing score convention, masks and flattening.
   Row-wise argmin is diagnostic; the actual Q-beam gate also needs flattened
   candidate top-K overlap/order, invalid-slot leakage, ties, best/second-best
   and K/K+1 margins, and downstream replay. Keep argmax explicitly auxiliary.
5. Keep categorical stress inputs for coverage and add legal scrambles or
   recorded frontiers for task quality. Array dtype/range validity is not
   puzzle reachability.
6. Record direct pairwise tensor differences. Matching aggregates against an
   oracle do not establish equality between two implementations.
7. Compare parameter delivery explicitly: runtime arrays versus captured
   constants are separate experimental conditions. Preserve the working BN
   path and do not extrapolate its measurements to the LN model.
8. Separate trace/compile/first execution, transfers, synchronized steady-state
   device work, full model, actual chunked caller and device scaling. Record
   device kind, runtime versions, shapes, tiling and timing scope. GPU warp,
   occupancy, shared-memory and stream heuristics are not TPU facts.

## Reconciled experiment order

The experts differed only in where they placed Q-beam checks in their written
sequence. Define score direction, masks, input classes and acceptance criteria
**before** running candidates. Establish JAX-only controls next, then attribute
Dense/LN differences and evaluate task metrics on every result. Time and promote
only the candidates that satisfy the predefined contract. No new universal
99% threshold is inferred from expert agreement.

## Regression scenarios for the planned plugin

- Opposite score direction and masked invalid outputs.
- A sole legal action, inverse-move exclusion, ties and near-ties.
- Correct row minima but different global candidate top-K.
- Segmented versus fused JAX-only reference boundaries.
- Reachable puzzle states versus categorical stress inputs.
- BF16 bias rounding and LayerNorm padding/statistic precision.
- Captured versus runtime weights under the same caller.
- Compile rejection, first-call/steady-state separation and microbenchmark
  improvement that disappears in the full model.

Each scenario should record its scope and expected observable behavior. Neither
interpretation nor a CPU check proves TPU compilation, communication safety or
performance. These recommendations extend the research basis for the plugin;
no plugin implementation, inference change or remote experiment is claimed here.
