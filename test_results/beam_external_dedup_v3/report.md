# External Stream3 dedup V3: physical correctness gate

Source `5e08ff60f3d470cef6ccdf0fc173510a827aecd3`, launcher `007f385`.
Kaggle COMPLETE, inspected on 2026-09-05. Both cases are exact, rather than
merely completing compilation. JAX/jaxlib 0.10.2, libtpu 0.0.42.1;
eight devices with IDs 0..7, all TPU v5 lite.

| Capacity per device | Mismatches | Compile s | Median ms | p10 / p90 ms |
|---|---:|---:|---:|---:|
| 256 | 0 | 1.67434 | 0.72310 | 0.69402 / 0.82041 |
| 512 | 0 | 1.53592 | 0.76190 | 0.698831 / 0.79793 |

Output and independent expected SHA-256 match in each case (see JSON).
The gate includes inclusive threshold, Hash128 then score/payload winner,
stable compaction, neutral padding and aligned survivor counts. Rank fixtures
include empty, full, partial, threshold-boundary, zero-hash and all-duplicate
inputs. Counts and metadata participate in the combined equality check.
This physically confirms the V2 scratch-layout correction, preserving semantics.

Each case has three warmups and 21 synchronized samples. These are diagnostic
whole-call latencies, not matched A/B against another implementation. Capacities
are not actual survivor counts; no useful-candidate throughput is inferred.
Two external bitonic sorts are a correctness baseline, not proof of efficient
large-frontier sorting. The different capacities do not establish scaling.

Scope ends before owner routing. No remote exchange, resident collector,
full depth, CUDA replay, inference acceleration or overlap claim follows.
Next gate must attach routing after dedup, preserve payload tie-breaking and
explicit counts, and test larger capacities plus collector overflow separately.
No shard-local top-k or cap may be introduced.

Artifacts: `beam_external_dedup/external_dedup.json`, both compiled HLO files,
and `tpu-beam-external-dedup-probe.log`. Runtime metric-server and huge-page
warnings did not prevent either case from compiling or executing.
