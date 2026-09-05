# Collector V2: two exact groups, full collector compile rejection

Kaggle terminal status ERROR; bundle `all_exact=false`. Source
`bad92c169a1001878ccb625f609c6cb634585b53`, JAX/jaxlib0.10.2,
libtpu0.0.42.1, eight TPU v5 lite devices, IDs0..7. Complete output and
top-level log downloaded from `trydotatwo/tpu-beam-collector-probe`.

| Isolated group | Correctness | Median ms | p10..p90 ms |
|---|---|---:|---:|
| single, capacity256/input128 | zero mismatches, output hash=expected | 0.660301 | 0.639590..0.727800 |
| group, capacity512/input512 | zero mismatches, output hash=expected | 0.615211 | 0.573780..0.633941 |
| full, three shards/rank | compilation rejected, not executed | unavailable | unavailable |

Accepted groups each use3warmups/21synchronized samples. These separate
processes/workloads are NOT matched A/B and do not show inference/beam speedup,
residency or overlap. Full group produced no timing or output comparison.
All JSON, process logs and the two successful HLO files are retained.

## Concrete failing expression

Full group's compiler reports E2003 unproven memory-access alignment at
`beam_collector.py:304`: `source = o[0,pl.program_id(0)]+relative`.
The operation is a VMEM `memref<1x128xi32>` dynamic column load returning
`vector<1x1xi32>`; the compiler cannot prove that column is128-aligned.
This is a handled compilation exception (child returncode1), not native abort.

A new structural JAXPR regression fails on exactly `[a:u32[] <- b[0,c]]`.
The proposed fix reads the aligned offsets vector and selects/reduces its
matching shard lane. Offsets are bounded by incoming capacity<=16384, so
signed32 reduction is exact. This removes the dynamic scalar access; physical
confirmation is still required, and further compiler failures remain possible.
Do not submit the prepared integrated S3/RDMA/collector gate until this full
collector has passed its next physical gate.
