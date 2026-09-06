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

Further source inspection: the top-level CUDA library includes Stream1 and
dispatcher even for final tests. Configuration requires CUTLASS and NCCL
(`find_path(nccl.h REQUIRED)` and `find_library(nccl REQUIRED)`). Therefore
finding Windows nvcc alone is insufficient for this CMake route. Current PATH
has CMake and nvcc but not `cl`; this does not prove MSVC is uninstalled, only
that a compiler environment has not been established. No configure/build was
attempted. Prefer checking an isolated Linux CUDA environment before treating
the source's full build as locally available; do not remove its NCCL dependency
or modify source files to make a weaker test appear equivalent.

Read-only Docker inventory (`docker ps`, `docker image ls`) failed because
the `dockerDesktopLinuxEngine` named pipe was absent. No container/image
inventory or container GPU visibility was obtained. Docker/WSL was not
started, stopped or reconfigured. This limits the currently available local
CUDA execution route; it does not block independent TPU implementation/tests.
