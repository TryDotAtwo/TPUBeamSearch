# V5 isolation results

Source7cabf39e866642ccd1f5d1b9993601faa3b4e3af; launcherbace33c.
All six subprocess reports identify JAX0.10.2/libtpu0.0.42.1 and eight devices.
Artifacts: `test_results/beam_final_v5_isolation/beam_final_isolation`.

| Scope | Return code | Result |
|---|---:|---|
| DMA | 0 | Both count0/2 exact on8devices |
| Cast round-trip | 0 | Both count0/2 exact on8devices |
| Gather1D | -6 | Native layout assertion during count0 compile |
| Gather2D | 1 | Unsupported gather lowering before backend compile |
| Packing | -6 | Native abort during count0 compile |
| Production | -6 | Native abort during count0 compile |

Successful cases have eight zero mismatch counts and each output SHA matches
the expected SHA. The count2 cases exercise actual reads; count0 alone would
not. The cast round-trip may be optimized away, so it does not independently
prove execution of conversion instructions.

The extracted one-dimensional gather path is a sufficient reproducer for the
layout.h341 assertion(1versus2). DMA alone succeeds with identical record-axis
storage. This narrows the failure beyond a stack-only guess but does not prove
that all production problems have a single cause. The attempted2D variant
fails with `Only take_along_axis-like gathers supported`; it is not a working
alternative. Packing cannot be separately blamed because it includes gather.

Lowered MLIR is preserved before compilation. Its TPU custom-call body is
serialized, so merely searching the outer text is not a decoded internal-IR
analysis. No timings, physical-allocation measurements, full-beam or speedup
claims follow from this run.

Next diagnostic should compare a supported gather representation with an
integer equality/select/reduction permutation fallback, holding input bytes
fixed. The fallback is a correctness/compilation experiment, not a performance
recommendation. Keep production unchanged until physical evidence is obtained.
