# Artgor Exact Accelerated TPU Notebook Design

## Goal

Publish a user-ready Kaggle TPU notebook derived from Andrey Lukyanenko's
`cayleypy-cube555-tpu-beam-q` notebook at script version `344319112`. Preserve
its default checkpoint, puzzle assets, beam-search semantics, endgame handling,
path replay, configuration surface, and result artifacts while replacing the
default Q-only inference path with the measured exact split engine from
TPUBeamSearch.

The public claim is intentionally narrow: the selected inference engine has
already measured approximately `1.6x` steady-state throughput over the original
monolithic JAX forward on eight TPU v5e devices at the target 32K local batch.
The notebook will report a separate end-to-end beam-step or solve speedup only
after measuring it. It will not present the inference result as an unmeasured
whole-solver speedup.

## Frozen source and model contract

The notebook is based on the immutable Kaggle source version linked by the
user, not an unspecified latest version. The retained default model contract
is:

```text
state                 uint8[B, 150]
embedding             BF16[150, 24]
flattened encoding    BF16[B, 3600]
input block           Dense(3600, 1024) -> LayerNorm -> ReLU
residual stack        10 * {
                        Dense(1024, 1024) -> LayerNorm -> ReLU
                        Dense(1024, 1024) -> LayerNorm
                        residual add -> ReLU
                      }
Q head                Dense(1024, 30)
score direction       lower Q is better
```

The default checkpoint remains `q555_2k_BEST.pt`. Model weights and outputs use
the original BF16 inference contract. The source notebook's move order,
symmetry transforms, inverse-frame handling, history logic, exact endgame,
packed backpointers, and path verification are unchanged.

## User-facing notebook

The accelerated notebook retains the original six-cell narrative and workflow
as closely as practical. The configuration cell gains one option:

```python
INFERENCE_ENGINE = "exact_split"  # or "original_jax"
```

`exact_split` is the default for the supported single-checkpoint, Q-only path.
`original_jax` remains an explicit correctness oracle and operational fallback.
If `BLEND_CHECKPOINTS` is enabled or `QV_CONSISTENCY != 0`, the notebook prints
a clear explanation and selects `original_jax`; it never silently runs an
unvalidated approximation under the accelerated engine's name.

Startup output records the notebook source version, TPUBeamSearch Git commit,
JAX/libtpu versions, TPU device inventory, checkpoint SHA-256, selected engine,
and effective inference/parent chunk sizes. A preflight rejects incompatible
device counts, shapes, dtypes, or chunk divisibility before compilation.

## Source distribution

GitHub remains the source of truth. The repository contains:

- the reusable exact inference implementation;
- the staged beam-step integration;
- a deterministic notebook builder based on the frozen Artgor notebook;
- the generated public `.ipynb` and Kaggle metadata;
- tests that validate both source modules and generated notebook cells.

Private validation kernels clone a pinned public Git commit. For the public
release, a deterministic packaging script exports only the required Python
package and a manifest containing that commit into a versioned public Kaggle
dataset. The notebook attaches this code dataset alongside the original Artgor
artifact dataset and competition dataset, keeps internet disabled, imports the
package from the mounted dataset, and verifies the manifest commit at startup.
This makes Copy & Edit self-contained while GitHub remains the auditable source
of truth. Secrets and unrelated local artifacts are never included.

## Exact split inference engine

The accepted engine uses two separately compiled, device-resident dispatches:

1. prepacked Pallas flat embedding with `BM=4096`, followed by the original
   JAX/XLA input Dense, all LayerNorm operations, and ten residual blocks;
2. Pallas `1024 -> 30` head with `BM=256`, `BK=1024`, `BN=128`, and late BF16
   rounding.

The hidden tensor passes directly from prefix to head on TPU. No host
materialization or `device_get` is allowed. The pair is not enclosed in an
outer `jax.jit`: preserving the real dispatch boundary is part of the numerical
contract. Earlier experiments showed that apparently equivalent fused or
materialized boundaries can change the final-residual Dense schedule and BF16
outputs.

The initial integration uses a local inference chunk of 32,768 parents per TPU
core because that exact shape passed the frozen full-Q gate and measured about
`1.62x` over original JAX. One chunk's BF16 hidden tensor is 64 MiB per core,
which avoids the roughly 4 GiB hidden tensor that would result from evaluating
the full 2,097,152-parent local beam at once.

## Staged beam-depth data flow

The original notebook embeds inference inside one compiled `shard_map` beam
step. That form cannot host the accepted engine because an outer compilation
may remove its required prefix/head boundary. The accelerated path therefore
splits one depth into explicit device-resident stages while keeping the search
ordering unchanged.

For the default `B_LOCAL=2,097,152` and `PARENT_CHUNK=131,072`:

1. Slice each of the 64 ordered 32,768-parent inference chunks directly from
   every core's state shard inside a prefix dispatch.
2. Run the separate Pallas head dispatch immediately after each prefix and
   retain only its BF16 Q output; no chunk hidden tensor is retained.
3. Assemble the ordered Q chunks into one device-resident
   `[8, B_LOCAL, 30]` tensor. This costs about 120 MiB per core, versus roughly
   4 GiB per core for a full-beam hidden tensor.
4. Invoke one search dispatch containing the original 131,072-parent
   `lax.scan`. Each scan iteration slices its Q values instead of calling the
   model, then performs the unchanged child generation, inverse mask,
   hash/owner routing, per-destination running top-K, `all_to_all`, history and
   dedup, final selection, solved/endgame checks, and packed backpointers.

Parent-window size and scan order remain 131,072 even though inference uses
smaller chunks. This preserves the existing streaming-selection and tie
boundaries. Q, selected states, owners, incoming moves, and packed
backpointers remain on TPU between stages. Only the existing per-depth
backpointer copy and progress metadata cross to the host.

The first production implementation queues only the natural prefix-to-head
dependencies and does not add a second host/device streaming protocol, custom
overlap, or larger chunks. Those are later optimizations and must independently
pass the same exactness gates.

## Numerical and search exactness gates

The accelerated path is accepted only if all comparisons are bitwise exact;
argmin or top-K agreement alone is insufficient.

### Inference gate

At 32,768 states per device on eight TPU devices, compare original JAX and the
accelerated path on deterministic legal scrambles and categorical stress:

- every Q value finite;
- zero unequal BF16 elements;
- identical full output SHA-256;
- identical minimizing action and ordered top-K diagnostics.

### One-depth gate

Starting from identical sharded parent states and carry tensors, compare the
original and staged beam depth for:

- Q scores in parent/move order;
- generated children and inverse masks;
- hashes and destination owner IDs;
- per-owner selected values and original flat IDs;
- post-routing and post-dedup survivors;
- final state order, incoming moves, minimum-Q log, solved metadata, and packed
  backpointers.

Every compared tensor must have zero mismatches and the same hash. A compile
failure is recorded as a rejection, not as a timing result.

### Solver gates

Run a deterministic small-beam short-depth A/B and require identical frontier
and backpointer output at every completed depth. Then run the accelerated
notebook on the real default-width geometry for `pid=1034`, symmetry frame 0,
without inversion. The publication run must find a solution, and its path must
replay from the original competition state to the solved state, including an
independently verified exact-endgame tail.

## Performance protocol

Compilation, first execution, steady-state execution, and host orchestration
are reported separately. Synchronized timings record:

- prefix and head dispatches;
- inference assembly;
- child generation, hashing, and per-owner selection;
- `all_to_all`, dedup, and final top-K;
- backpointer transfer/write;
- complete beam depth;
- complete validated solve.

The original and accelerated engines use identical inputs and geometry in
paired runs where runtime permits. The notebook may state the already verified
inference result only with its exact measured batch/device scope. A new
end-to-end speedup is published only from a successful same-runtime comparison;
otherwise the report states that end-to-end improvement remains unverified.

## Error handling and fallback

- Unsupported accelerated modes fail over to `original_jax` before compilation
  with an explicit message.
- Shape, dtype, checkpoint, device, and divisibility mismatches fail fast.
- A failed exactness gate prevents publication of the accelerated engine as the
  default.
- TPU compile/runtime errors are checkpointed with the active Git SHA and
  configuration; they are not converted into fabricated latency rows.
- A path that fails host replay is discarded exactly as in the original
  notebook.
- Kaggle output includes a machine-readable validation JSON and a concise human
  summary beside the original solve/submission artifacts.

## Publication gate

The notebook becomes public only after:

1. generated-notebook JSON and Python-cell AST validation pass;
2. the focused integration tests and complete project test suite pass;
3. an eight-TPU private run passes inference and one-depth exactness;
4. the short solver A/B passes;
5. the real-width `pid=1034`, frame-0 run finds a path and passes independent
   replay;
6. the validation report clearly separates inference and whole-solver timing.

The public GitHub commit, Kaggle notebook slug/version, runtime inventory,
checkpoint/source/input hashes, results JSON, useful logs, and report are
recorded in `test_results/`.

## Non-goals

- Rewriting the full network in Pallas in this release; that remains the next
  research phase after the immediately usable notebook.
- Changing beam width, search quality heuristics, checkpoint selection,
  symmetry policy, endgame depth, history semantics, or move ordering.
- Accelerating blends or the auxiliary QV-consistency head without a separate
  exact implementation and gate.
- Claiming a 1.6x full-solver improvement from inference-only measurements.
- Fusing the exact prefix and head before a new bitwise and performance study.
