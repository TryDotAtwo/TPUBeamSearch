# Beam primitive TPU gate V1

Private Kaggle `trydotatwo/tpu-beam-primitive-compile-and-correctness-gate`,
source c05b269a09edc38c846840d2fb433848b83a7986. COMPLETE with `all_exact=false`.
Runtime JAX/jaxlib 0.10.2, libtpu 0.0.42.1, eight physical TPU v5 lite devices.

Five exact cases: serialized pack b2/b3, pipelined pack b2/b3, and Hash128
owner/shard routing. Zero differing integer elements across all eight replicas.
Six compile errors: both hash/goal shapes and all four diagnostic dedup cases.
There were no executed mismatching cases; rejected cases did not execute.

Hash/goal fails Mosaic gather lowering because the primitive uses CLIP mode.
Dedup fails because jnp.take produces index rank unlike take_along_axis. Compiler
messages: `Unsupported gather` and `Only take_along_axis-like gathers supported`.
The runtime's gather lowering requires elementwise indexed gather with compatible
batch axes and FILL_OR_DROP or PROMISE_IN_BOUNDS mode, not CLIP.

V2 fix: express clipping explicitly before promise-in-bounds rank-one gather;
use broadcast per-row indices for dedup take_along_axis. Structural JAXPR
regression reproduced both V1 failures and passes after the fix. CPU source parity
also passes, but only a new physical TPU run can validate complete lowering.

V1 deliberately collected NO steady-state timings. It cannot support a speedup
or an overlap claim. Successful HLO and complete logs are archived here.
