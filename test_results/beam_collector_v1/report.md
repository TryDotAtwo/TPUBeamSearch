# Collector V1: TPU lowering rejection

Private kernel `trydotatwo/tpu-beam-collector-probe` terminated ERROR.
Source: `6d303eb2698b15e6a2bce925c3adc141735bafa8`.
Runtime: JAX/jaxlib 0.10.2, libtpu 0.0.42.1, eight TPU v5 lite devices.

The saved JSON contains input/reference hashes, but no executed correctness
result, compiled HLO, or timing samples. No TPU correctness/performance claim.

Failure occurs in Mosaic `_gather_lowering_rule` during `fn.lower`, with
`NotImplementedError: Only take_along_axis-like gathers supported`.
The collector uses `jnp.take` to shift an incoming tile into an unaligned
destination. A local JAXPR regression reproduces the rejected shape contract:
operand/output `[8,128]`, indices `[128,1]` rather than `[8,128,1]`.
The same issue exists in the newly added functional multi-shard scatter
(`[8,512]` operand, `[8,128]` output). Both tests fail before changing code.

The candidate fix broadcasts indices across the eight metadata planes and
uses `jnp.take_along_axis` without changing index masking or record semantics.
Local shape-contract/interpreter checks cannot establish that all subsequent
TPU lowering stages accept the kernel; a new physical gate remains required.
The old V1 is terminal, not queued/running, and is not a benchmark slowdown.

Local final regression after the fix and collector bundle preparation:
664 passed in 435.19 s, no skips/errors/failures, source CPU oracle enabled.
See `../local_collector_bundle_regression.xml`. Physical validation remains
pending a new pinned run; this local pass does not change V1's failed result.
