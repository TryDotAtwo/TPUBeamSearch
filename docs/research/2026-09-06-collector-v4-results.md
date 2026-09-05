# Collector V4: physical eight-TPU acceptance of bounded gates

Private kernel `trydotatwo/tpu-beam-collector-probe`, V4, COMPLETE.
Source `147361e6b61e22cddbba22be9c3275a1b3b5c755`, launcher `d461dff`.
Artifacts: `test_results/beam_collector_v4/`, including both process logs,
compiled HLO, nested reports, coordinator report and full Kaggle log.
JAX/jaxlib 0.10.2, libtpu 0.0.42.1; device IDs0..7, TPU v5 lite.
Coordinator reports all_exact=true; both subprocesses returned0.

| Gate | Compile seconds | Median ms | p10/p90 ms | Mismatches |
|---|---:|---:|---:|---|
| Functional collector, three shards/rank | 1.208 | 0.718910 | 0.674110 / 0.790370 | 0,0,0,0 |
| Bounded128 S3 + snapshot RDMA + functional collector | 4.324 | 0.913981 | 0.880520 / 0.952681 | 0,0,0,0 |

Each uses3 warmups and21 synchronized repeats. These are separate processes
and different workloads, not matched A/B. No relative speedup is inferred.
Integrated fixture intentionally includes fatal admission on ranks3 and5;
actual flags exactly match `[0,0,0,1,0,1,0,0]`. Its latency is not beam throughput.

Full collector input SHA matches V3:
`24b304f4b9a47e55c62ba4ab3eb9d494adbb669b5903d0aeec7c419010ed67e7`.
Expected and actual output SHA:
`0e8cceecd17b09ba0444d96e71a8027f65a2a73cf17450bc82c1af09b546e0c3`.
Integrated expected and actual output SHA:
`6602a2e51a42e4b16c598ee7839abcda722cf715e7599039bf22152b6dca9a2c`.
Fixture provenance is original CPU C++ S3/routing plus source-audited Python
admission, with file hashes and dirty source state recorded. It is not CUDA.

The banked128 gather repair now physically compiles and passes this fixture;
V3's multi-vreg gather rejection is absent. This does not establish arbitrary
capacity scalability. Runtime metric-port/hugepage warnings did not prevent
compilation or correctness and are preserved in logs.

Remaining: persistent resident lifecycle, scalable HBM sort/collector,
coordinated S5, K1/K2 physical validation, final/history, whole multi-depth
GPU/8-TPU replay and measured overlap. The next prepared gate is the isolated
S4 reservation/publication, S5 request MAX and histogram SUM bundle. It is not
yet an integrated S5 epoch. No defaults or BN path changed.
