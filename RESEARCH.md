# Kaggle TPU and fixed-width beam search

Status: initial research, 2026-08-27. The repository was empty when this note was created.

## What Kaggle currently provides

- Kaggle Notebooks exposes a free **TPU v3-8** accelerator. TPU runs have a 9-hour execution limit; the documented TPU VM host has 96 CPU cores and 330 GB RAM. Availability is quota/queue dependent.
- A v3-8 is one single-host slice with **8 TensorCores (TPU cores)**: four v3 chips, two TensorCores per chip. A v3 chip has 32 GiB HBM2 and 900 GB/s HBM bandwidth, so the slice has 128 GiB aggregate HBM, physically distributed across eight cores (16 GiB per core).
- Each v3 TensorCore contains two 128x128 matrix multiply units plus vector and scalar units. BF16 inputs accumulate in FP32. It is a matrix-first accelerator, not a CUDA-compatible general-purpose GPU.
- Kaggle uses a TPU VM, so datasets are mounted in the notebook filesystem like CPU/GPU sessions. Current Kaggle TPU images include JAX/JAXlib; the exact versions must be printed in every run because the image changes.

Sources:

- Kaggle notebook hardware and limits: https://www.kaggle.com/docs/notebooks
- Kaggle TPU VM announcement: https://www.kaggle.com/product-feedback/369338
- Cloud TPU v3 specifications: https://docs.cloud.google.com/tpu/docs/v3
- TPU architecture: https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
- Kaggle TPU image releases: https://github.com/Kaggle/docker-python/releases

## TPU versus GPU for this workload

| Property | Kaggle TPU v3-8 | Kaggle 2x T4 GPU |
|---|---|---|
| Programming model | JAX/XLA (preferred here), TensorFlow, PyTorch/XLA | CUDA, CUB/CCCL, NCCL, PyTorch |
| Strongest operation | Large dense BF16 matrix multiplication | General SIMT kernels plus Tensor Cores |
| Memory | 8 distributed HBM shards; 128 GiB aggregate, 16 GiB/core | 2 independent 16 GiB GPUs; 32 GiB aggregate |
| Irregular control/data structures | Must normally be expressed as static bounded arrays and masks | Native kernels, atomics, scans, radix sort, dynamic counters are more flexible |
| Device communication | Fast TPU inter-chip interconnect and XLA collectives | PCIe/NCCL on Kaggle T4x2 |
| Custom low-level kernels | Pallas exists, but v3 support/performance must be verified | Mature CUDA ecosystem |

The aggregate TPU memory is attractive, but it is not one flat 128 GiB address space. A beam must be explicitly sharded, and global selection/dedup introduces collectives.

TPU is likely to win only if model scoring dominates and has large, TPU-friendly matrices/batches. GPU is likely to win if the loop is dominated by small models, integer state transforms, hashing, exact deduplication, irregular compaction, or global ranking.

## TPU-compatible beam-search shape

Use a fixed-capacity, structure-of-arrays representation throughout:

- `state[B_local, state_words]`: compact integer or low-precision state;
- `score[B_local]`, `state_id[B_local]`;
- `parent[B_local]`, `move[B_local]` backpointers;
- `valid[B_local]` mask and scalar valid count;
- preallocated candidate arrays of shape `[B_local, branching, ...]` or flattened `[C_local, ...]`;
- double-buffered current/next frontier.

Compile multiple depth steps into one `jax.jit` computation using `lax.scan`, `lax.fori_loop`, or `lax.while_loop`. Loop-carried arrays must keep identical shapes and dtypes. Never bring scores/counts to Python inside the hot loop.

Candidate generation should be vectorized (`vmap`, gathers, scatters, `where`) and padded to a static capacity. Invalid and duplicate entries receive `-inf` score rather than changing tensor shape.

### Selection

Start with exact `jax.lax.top_k`. For eight-way sharding:

1. Each core expands and scores its local beam shard.
2. Each core selects a local oversampled top-k.
3. Exchange only `(score, state_id, parent, move)` summaries.
4. Select the global beam from the merged summaries.
5. Gather full state payloads only for survivors and redistribute the next beam.

`jax.lax.approx_max_k` is worth a separate experiment for huge candidate axes, but it is not a drop-in correctness-preserving replacement: recall is approximate and tie behavior is not stable. It cannot be the default for replay-exact search.

Modern JAX favors `NamedSharding`/`PartitionSpec` with `jit`, or `shard_map` when collectives must be explicit. `pmap` still works but is no longer the preferred starting API.

References:

- Static loop carry: https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
- Exact top-k: https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html
- Approximate max-k: https://docs.jax.dev/en/latest/_autosummary/jax.lax.approx_max_k.html
- Distributed arrays: https://docs.jax.dev/en/latest/201/sharding.html
- Explicit SPMD: https://docs.jax.dev/en/latest/notebooks/shard_map.html

## Hard part: deduplication and history

Exact global visited-set semantics are the main TPU risk. A mutable hash table with data-dependent occupancy is a natural CUDA/CPU design but a poor first XLA design.

Prototype in this order:

1. No deduplication: establish scoring and selection throughput.
2. Parent/move backpointers only; materialize paths on the host after completion.
3. Local beam dedup by sorting compact `state_id` keys, masking adjacent duplicates, and selecting valid rows.
4. Bounded recent-history dedup using fixed-size tables or sorted windows.
5. Only then test global persistent history. Keep a correctness-preserving overflow/fallback path; never silently drop states.

If exact persistent global dedup dominates runtime or cannot be expressed without large all-to-all traffic, use a hybrid: TPU for batched scoring, CPU for history. This has a high host-transfer cost, so it is a fallback to measure, not the preferred design.

## Minimum honest experiment

Implement one algorithm in NumPy CPU reference and JAX, then run identical seeded inputs on Kaggle TPU v3-8 and GPU. Do not compare against the existing native CUDA solver until semantic equivalence is established.

Correctness gates per depth:

- exact selected `(score key, state_id, parent, move)` sequence under a defined tie-break;
- same valid counts and overflow flags;
- replay every reconstructed solution on CPU;
- hash each frontier so long-run divergence is localized to a depth;
- exact top-k first; approximate selection is a separate quality experiment.

Benchmark phases separately and end-to-end:

- expand/state transform;
- neural scoring;
- local top-k;
- global merge/collectives;
- dedup/compact;
- full depth step and multi-depth compiled loop.

Sweep at least:

- beam: `2^12, 2^16, 2^20` (larger only after memory preflight);
- branching factor: representative small/medium/large values;
- model: no model, small MLP, production-sized scorer;
- one TPU core versus all eight;
- exact top-k versus oversampled hierarchical top-k;
- dedup off/local/global.

Record compilation separately from steady-state execution. Warm up once and use `.block_until_ready()` for every timed JAX result because dispatch is asynchronous. Report candidates/s, steps/s, end-to-end wall time, memory per core, compile time, and communication share.

Benchmarking reference: https://docs.jax.dev/en/latest/async_dispatch.html

## Immediate implementation direction

The safest first milestone is a synthetic JAX kernel with fixed state width, deterministic move transforms, an MLP scorer, exact hierarchical top-k, and parent pointers. It should run unchanged on CPU/GPU/TPU backends and emit a compact JSON benchmark plus frontier hashes. Only after that should puzzle-specific state transforms and exact history be added.

The go/no-go criterion should be based on end-to-end step time:

- proceed with TPU specialization if scoring plus selection scales across eight cores and remains the majority of runtime;
- stay with CUDA if dedup/state transforms/global selection dominate or TPU padding/collectives erase the scoring gain;
- consider a split backend only if host transfers can be amortized over several compiled depth steps.
