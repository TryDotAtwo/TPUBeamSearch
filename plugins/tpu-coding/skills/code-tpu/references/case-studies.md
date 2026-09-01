# TPUBeamSearch: scoped evidence, not defaults

Audited 2026-08-31 and updated 2026-09-01. Historical runs used JAX 0.10.2; local arithmetic witnesses
used CPU JAX 0.10.1. Some old reports inconsistently name TPU generations and
JSON lacks `device_kind`. Keep that uncertainty; do not derive a hardware peak
from the prose label. Evidence IDs below refer to [evidence.json](evidence.json).

## Two distinct families

| Contract | BN categorical MLP | Artgor LN ResMLP |
|---|---|---|
| State logical / storage | 120 / 128 | 150 / 150 in the measured harness |
| Input | 120 classes, virtual one-hot 14400 | 150 classes, embedding dimension 24, flatten 3600 |
| Trunk | 1536→512, 2 residual blocks at 512 | 1024, 10 residual blocks at 1024 |
| Normalization | inference BN folded offline | 21 runtime LNs, epsilon 1e-5 |
| Head | 24 move scores per parent | 30 move scores per parent |

Embedding `[150,24]` does not mean 24 categories. Scalar-V capability smoke
and incomplete-prefix timings are neither full Q model. The LN checkpoint also
has an auxiliary value head that Q-only inference does not execute.
[Architecture audit](https://github.com/TryDotAtwo/TPUBeamSearch/blob/f17eedff869f2cb23535c99b63ae024c6aa602cc/docs/research/2026-08-31-tpu-coding-research.md).

## Measurements worth retaining

| ID | Measured observation | Limit on inference |
|---|---|---|
| M-BN-PREFIX | Full BN model: 3.313M states/s, local B32768; prefix BM1024/BK128/BN1536 | Prefix-only BK256 winner was not the full-model winner. |
| M-BUFFER | Old prefix tile: default 2.799ms, explicit 1/2/lookahead 2.774/2.840/2.961ms | No demonstrated manual buffering benefit or profiler-established cause. |
| M-BN-SCALE | 8 independent devices: 25.482M states/s, 96.04% weak-scaling efficiency | Local B32768, no communication/search stages. |
| M-LN-INPUT | B16384: JAX embedding prefix 7.021ms; Pallas gather 8.369ms; virtual one-hot 11.166ms | BN's one-hot preference did not transfer. Folded variant was compile-rejected, not slow. |
| M-LN-FULL | B16384: original 1.386M; Pallas separate/per-layer/per-block .577/.547/.565M states/s | Full-model Pallas did not win; fixed BM128 comparison is not best-per-boundary tuning. |
| M-LN-VMEM | 8/32 screened candidates requested 16.06–16.36MiB against 16.00MiB scoped limit | Per-block BM256 rejection; not a physical-memory inventory. |
| M-LN-EXACT-FRONTIER | Eight TPU v5 lite, local B32768: exact hybrid 15.363/15.364ms legal/stress versus original JAX 24.865/24.852ms | Exact split after the final residual block; residual trunk remains JAX/XLA. Head-only attribution is much smaller than total gain. |

Exact runs, JSON and source pins are linked in the evidence records.
**M-BN-FUSION:** BN prefix fusion gained about 2.2–2.7% in matched-tile
comparisons. **M-BN-RESIDUAL:** near-tied full-model residual fusion variants
favored simpler separate kernels; 95.3125% argmax agreement was diagnostic,
not minimizing-task quality, and accumulation-order causality was unproven.
These observations justify controlled A/B, not a universal fusion or encoding ban.

The exact frontier result changes the earlier engineering conclusion without
rewriting the old failed all-block Pallas sweep. Preserving the JAX residual
arithmetic while changing the execution boundary and Pallas embedding tile
produced the robust gain. A Pallas head then passed the unchanged exact gate,
but its composed delta over the same BM4096 prefix with a JAX head was only
about 0.14–0.33%, and its standalone 32K timing was slower. Promote the full
two-stage configuration; do not generalize that every replaced operator wins.

## Corrected interpretations

- **S-JIT:** depth-1 hidden mean error .000545 and final argmax agreement .7301
  were compared across different JAX compilation boundaries. Missing same-suffix
  JAX control prevents causal attribution to Pallas perturbation amplification.
- **S-RANK:** the original Q consumer minimizes; published argmax agreement is
  an auxiliary metric, not agreement of the chosen move or global frontier.
- **H-ROUND:** BF16 dot-result/bias rounding differs at expression level. A CPU
  witness gives max abs .03125; contribution on the target TPU is unmeasured.
- **D-PAIR:** equal error summaries against an oracle are not direct pairwise
  equality. Preserve original aggregates but correct that conclusion.
- **D-INPUT:** hashed `[0,150)` categorical rows are not necessarily permutations
  or reachable puzzle states. Legal scrambles and actual frontiers are still needed.
- **M-CAPTURE:** runtime weights fixed one BN captured-constant VMEM failure.
  Both sides of the LN full-model A/B capture weights; one-sided capture is not
  the established explanation for that comparison.

The [published audit](https://github.com/TryDotAtwo/TPUBeamSearch/blob/f17eedff869f2cb23535c99b63ae024c6aa602cc/docs/research/2026-08-31-tpu-coding-research.md)
supersedes those interpretations without rewriting raw measurements. The
[expert follow-up](https://github.com/TryDotAtwo/TPUBeamSearch/blob/f17eedff869f2cb23535c99b63ae024c6aa602cc/docs/research/2026-08-31-tpu-expert-followup.md)
supports the diagnostic plan as peer advice, not independent TPU measurement.
