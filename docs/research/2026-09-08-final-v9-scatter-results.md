# V9: physical scatter and local chain accepted

Kernel `trydotatwo/tpu-beam-final-gate` V9 completed. Source
`ef293af3753f06b6d031a1dd35b7d9d1df034501`, launcher `1299461`.
All output and logs downloaded to `test_results/beam_final_v9_scatter`.
Both subprocesses returned zero. The local independent fixture verifier
accepted both nested reports, not merely their all_exact flags.

| Mode | Cases | Result on each of eight devices |
|---|---|---|
| scatter | counts 0, 1, 127, 128, 129 | zero byte mismatches, matching SHA, zero errors |
| scatter | count overflow, target overflow | frontier unchanged byte-for-byte, error count 1 |
| materialize then scatter | counts 0, 1, 127, 128, 129 | zero byte mismatches, matching SHA, zero errors |

Both reports identify eight distinct TPU v5 lite devices, JAX/jaxlib 0.10.2
and libtpu 0.0.42.1. All twelve cases executed and have lowered MLIR and
compiled HLO artifacts. Input/expected/output hashes were independently checked
against pinned local fixtures, including canary rows and zeroed index tails.

The internal record-axis DMA layout resolves V8's demonstrated load alignment
rejection and allows both reading arbitrary records and writing arbitrary
target rows in these fixtures. External two-dimensional wire/frontier ABI is
unchanged. This does not prove zero-copy reshapes or general scratch donation.

Scope: identical local fixtures replicated across eight devices, not a routed
request/response exchange. No timing samples were collected. No inference,
beam speedup, overlap, full frontier/history publication or multi-depth replay
claim follows from this result.

Next: integrate common error agreement and routed response chunks with already
prepared rank grouping/intervals; preserve destination-capacity checks and
source/send/receive/consumer lifetime rules before any publication.
