# Controlled BF16 prefix capture

Inference-only follow-up to `test_results/artgor_prefix_shape_v2/report.md`.
The unchanged large-shape JAX oracle remains mandatory. No speed promotion.

Use the identical legal42 corpus, 131072 states partitioned over eight TPU.
Enforce its previous input SHA and reproduce both untouched large output SHAs.
Run at 16384 and 256 rows/device, preserving device-major chunk reconstruction.

Capture BF16 Dense, broadcast BF16 mean, and output together for JAX and Pallas.
Compare captured output to each untouched executable. Extra outputs can change
fusion: a failing capture control forbids attribution to the untouched oracle.
Record Dense/mean/output shape comparisons, not just output differences.

Cross JAX/Pallas materialized Dense and mean through the same Pallas remainder.
The Pallas/Pallas substitution must reproduce the untouched Pallas prefix.
The other three substitutions remain diagnostic if either capture or zero-change
control fails. In particular, matching a substituted output alone proves nothing
about the arithmetic of an invalid captured reference.

Save finite, bitwise/numerical/signed-zero counts, complete SHA-256, affected
state/output rows in NPZ, and compiled HLO/StableHLO for captures, controls and
the shared remainder. CPU tests validate the capture's dtype/slots only; TPU
execution decides whether compiler instrumentation is non-perturbing.
