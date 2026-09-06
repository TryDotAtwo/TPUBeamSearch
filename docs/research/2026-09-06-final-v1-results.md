# Final gate V1: partial physical acceptance

Source `0e98b90310bc897a941f61bbd5e5bf2cbccfc9c7`, launcher `308e156`.
Artifacts: `test_results/beam_final_v1`, including original nested reports,
process logs, HLO and Kaggle log. Runtime: JAX/jaxlib 0.10.2, libtpu 0.0.42.1,
eight TPU v5 lite devices, IDs 0 through 7.

| Group | Return code | Exact cases | Outcome |
|---|---:|---:|---|
| CUDA fixture materialization | 1 | 0/6 | First case failed compilation |
| Final exchange | 0 | 16/16 | All reported outputs exact |
| Final coverage agreement | 0 | 7/7 | All reported outputs exact |

The isolated coordinator continued after materialization failed. The complete
bundle is correctly marked `all_exact=false`. No materialization numerical
result exists; five later CUDA fixture cases were never reached.

## Rejection and candidate correction

`count0_remote` fails at `beam_final_materialize.py:49`: the pipelined output
BlockSpec `(1,128)` for an array `(128,128)` violates the TPU requirement that
the penultimate block dimension be divisible by eight or equal the entire
corresponding array dimension. This is a Python lowering ValueError, not a
native compiler abort, numerical disagreement, or performance measurement.

A new structural regression test traces the actual output mapping and failed
on `(1,128)` before the correction. The candidate uses an explicit HBM output
and serialized row DMA. Parent staging is reused only after the parent-read
wait; every output transfer is waited, including zeroed invalid rows.
Two focused tests passed in 7.32 seconds. Full local regression: 936 passed in
1210.32 seconds with both C++ oracle paths, zero failures/errors/skips; see
`test_results/local_final_dma_output_full.xml`. These tests do not establish
physical TPU compilation acceptance of the correction.

## Scope limitations

Exchange tests cover repeated snapshots; coverage checks test common error
agreement. Neither proves a full frontier publication protocol or DMA drain.
There are no latency samples, overlap profiles, multi-depth beam results or
speedup claims here. Actual CUDA fixture bytes remain the materialization
oracle; local C++ tests are not additional CUDA executions. V2 must compile
and execute all six materialization cases before this gate can be accepted.
