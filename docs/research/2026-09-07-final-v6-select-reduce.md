# V6 select-reduce: physical correctness accepted for isolated permutation

Sourcebe939296d25133e20ae921ddb4547aaa4a8c457e, launcherf997334.
Runtime JAX/jaxlib0.10.2, libtpu0.0.42.1, eight TPU v5 lite devices IDs0..7.
Artifacts: test_results/beam_final_v6_select_reduce including full log,
lowered MLIR, compiled HLO and probe.json.

Both count0 and count2 execute. Each has eight zero mismatch counts and eight
output SHA256 values matching the expected SHA. The probe is exact, not merely
COMPLETE. Count2 exercises parent reads and permutation; count0 tests zeroing.

Integer equality-mask/reduction avoids the failing dynamic gather for this
isolated width128 fixture. Complexity is quadratic in width; no timing or
speedup is measured. This does not establish full materialization, request
validation, response packing, or full beam correctness. Production remains
unchanged pending TDD integration and the full physical CUDA-fixture gate.
