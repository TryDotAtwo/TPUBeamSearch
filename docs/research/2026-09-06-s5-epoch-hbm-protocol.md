# Explicit-HBM S5 composition gate

V6 accepted raw remote histogram wires on eight TPU v5lite devices, with
and without initialization. This protocol checks the next two composition
boundaries, not concurrent beam execution.

Run `benchmarks.beam_s4_s5_bundle --epoch-control --output <directory>`.
The coordinator creates no JAX client and runs sequential subprocesses:

1. `hbm_combined`: the eight existing histogram fixtures, width 256, remote
   exchange with an explicit HBM output allocation followed by Pallas pair
   reduction. Compare all uint32 elements and SHA-256 against host totals.
2. `epoch`: twenty serialized, state-carrying epochs at histogram width 128.
   Cover no request, each individual requesting rank, and all-rank request,
   twice. Feed actual device threshold slots, active slot and epoch state
   forward. Compare every output at every epoch, including inactive slots.

Continue to the second subprocess even if the first fails. Acceptance requires
both zero return codes and every nested case exact, with expected/output hashes
equal. Inspect pinned source, JAX/jaxlib/libtpu and all eight physical devices.
Preserve partial JSON, process logs and compiled HLO on failure. Native abort
does not identify a source expression without further isolation.

Histograms remain frozen for the whole epoch probe. Callers must drain S4
writers and threshold readers; no concurrent protection, ring overlap or full
beam speedup is established. Epoch execution is correctness-only. Primitive
timings, if present, are not matched across subprocesses.

The explicit-HBM option defaults to false in the composition factory; existing
production defaults and the BN path are unchanged. Public source must pass the
full local suite with both C++ oracle executables before pinning the launcher.
Local interpretation and C++ execution do not constitute TPU or CUDA evidence.
