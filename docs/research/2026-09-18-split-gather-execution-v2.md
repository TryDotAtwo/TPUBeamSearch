# Split-gather physical execution V2

Kernel: `trydotatwo/tpu-beam-split-gather-execution`.
Source `167c7ecfb77263b97d728bd0f667d8667076bfdf`, launcher `4db542b`.
Artifact: `test_results/split_gather_execution_v2/`.

The physical TPU gate completed on eight distinct TPU v5 lite devices. The
result JSON reports output shape `[8,32,128]`, dtype `uint32`, `exact: true`,
and `max_abs: 0` against the independent host oracle. The output is finite by
construction (unsigned integer dtype); 19 control lanes are nonzero for the
intentional interval metadata and are not payload mismatches.

This establishes execution and exact payload semantics for the isolated
split-gather diagnostic. It does not establish integration into production
packing, response epochs, inference, latency, overlap, or beam throughput.
