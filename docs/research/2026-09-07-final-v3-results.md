# Final V3: gather bitwidth rejection

Source090a81ec995423509129131969ca107f53f8091e, launcher18bcd4e.
Artifacts: `test_results/beam_final_v3` (nested reports, HLO and full logs).
Exchange16/16 and coverage7/7 remain exact; materialization fails compilation
in the first case, with five further cases unexecuted. No materialization
numerical result or timing exists.

Mosaic reports unsupported `dynamic_gather` of vector128xi8 using vector128xi32
indices. The preceding row-DMA alignment rejection is no longer reported;
this is not evidence of physical execution or zero-copy record-axis reshapes.

V4 minimally widens gathered data to int32, then narrows the selected values
back to uint8. All byte values are exactly representable in int32. The test
checks actual nested gather bitwidths and byte-exact output over all256 values
using reversed width512 states, logical length508, seven parents and zero tails.
The structural test failed before implementation. Four focused tests passed.

Initial full run83939 had937passed/1failure: another test module globally enabled
x64, promoting indices to64bits. The new test now scopes x64=False to match
the unchanged pinned TPU launcher and restores the incoming setting. Combined
tests10passed12.81s. Full76339:938passed1231.27s, zero failures/errors/skips,
both C++ oracle paths enabled. Both full-run XMLs are retained.

These are local regression results, not TPU compilation acceptance or CUDA
execution. V4 still requires the complete physical29-case gate. Full beam,
multi-depth GPU/TPU replay and performance remain unverified.
