# Local CUDA replay inventory (read-only)

Current nvidia-smi reports one NVIDIA GeForce RTX3070 Laptop GPU,8192MiB,
driver572.70. Get-Command locates nvcc under CUDA/v12.5/bin. No CUDA workload
was launched and no source checkout was modified by this inventory.

The read-only D:/100XH100/CMakeLists.txt defines architecture75;86 by default
and stream2/3/4/5, final, threshold, stitched, dispatcher and static-memory CUDA
test targets. This suggests a route to actual single-device differential
evidence using a build/output directory inside TPUBeamSearch, but buildability,
runtime execution and fixture compatibility are unverified.

The one visible device cannot establish multi-GPU communication correctness.
Do not relabel the already passing CPU C++ adapters as CUDA tests. Future
physical comparisons need identical metadata, scores, tie rules, capacities
and depth transitions with source/checkpoint/fixture hashes. A source test
target's existence or GPU visibility alone does not meet the replay gate.
