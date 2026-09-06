# Actual single-GPU source final smoke test

Executed on NVIDIA GeForce RTX3070 Laptop GPU,8192MiB, driver572.70.
Original `D:/100XH100/tests/final_cuda_tests.cu` was compiled with original
`cuda/final_materialize.cu` and `src/state.cpp`; no source files were changed.
Build and execution outputs remain inside `.local/cuda_final` in TPUBeamSearch.

MSVC Build Tools was found through vswhere and initialized with vcvars64.bat.
CUDA12.5 nvcc compiled successfully, without allow-unsupported-compiler, for
sm86, C++17, logical state120, physical128, alignment16 and MOVE_COUNT24.
Include directories were the source `src` and `cuda`. The targeted three-file
link does not depend on the full CMake library's NCCL/CUTLASS linkage.

Compiler exit0. Executable exit0, stdout `final_cuda_tests=pass`.
The original test's report contains:

```
response_target_idx_pack=pass
write_next_frontier=pass
padding_cleanup=pass
status=pass
```

The source report filename/title still says2026-05-20; execution was2026-09-06.
This is a one-request identity/swap smoke fixture. It checks target index0,
the first two next-frontier values and persistent zero padding. It does NOT
compare full GPU/TPU arrays, exercise inter-GPU communication, cover every
final function, measure speed or establish multi-depth beam replay.

## Source SHA256

| File under D:/100XH100 | SHA256 |
|---|---|
| tests/final_cuda_tests.cu | 748bed17ed24f7fafecd9577aee5c231413c0132fb93c36e747b8ad91ac08033 |
| cuda/final_materialize.cu | c3578668299fd2883ef9a0dda336e7bba1376f4ed44fad613a0867ad041585ea |
| cuda/final_materialize.hpp | 2a160351b26e88c4e8115947a03c04b920f059f2efbcdde2a10177142cd1b6c8 |
| src/state.cpp | 7094caed55c9876eb496a4e5c705b814309b5d3807b7e935712b78d09614406c |
| src/state.hpp | e8f5e54396bf617a36edf5ef41a0db5f992b63ef18a96a98826288db379db446 |
| src/types.hpp | 504c54cf72870da281d13e4a68871ccf1a4f061755d7acfdf456818cf7e98a5f |
| src/config.hpp | 91329b4d68b3e3fe6be2e88defdd90e5b293a22c7083391ee69a5ba67e04ba52 |

Next GPU evidence should use an external adapter calling these unchanged
CUDA functions on the SAME serialized fixtures as TPU and save all outputs.
That adapter is not implemented by this smoke test.
