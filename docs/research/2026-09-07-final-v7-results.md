# V7: separate final components accepted on eight TPU devices

Source7c5f5854b2d73f1f90a574dd8c2a35192ed8760a, launcher9d5ec02.
Full artifacts: test_results/beam_final_v7. All three subprocesses returned0;
the bundle reports29exact cases. Runtime JAX/jaxlib0.10.2, libtpu0.0.42.1,
eight TPU v5 lite devices.

| Component | Exact cases | Evidence |
|---|---:|---|
| Materialization | 6/6 | Zero byte mismatches and CUDA SHA on each device |
| Exchange | 16/16 | Repeated transfer snapshots match expected output |
| Coverage agreement | 7/7 | Expected common errors and coverage checks |

Materialization covers counts0,1,127,128,129 for remote-labelled CUDA fixtures,
plus127 local. Invalid counts are zero and all per-device hashes match the
published actual-CUDA fixture hashes. These fixtures execute independently on
each TPU: the materialization test itself is not distributed exchange.

The working path uses uint8 record-axis DMA plus integer equality/reduction
permutation. It avoids the native gather failure isolated inV5. The permutation
is O(width squared); neither speed nor zero-copy layout conversion is claimed.

This gate validates components separately. It does not validate integrated
final publication/drain, host history commit, shared scratch lifetime, or
multi-depth beam execution. There are no timing samples in this gate. Next:
compose the accepted components and verify ownership/publication/lifetimes,
then same-score GPU/8-TPU replay, with actual inference and profiling separate.
