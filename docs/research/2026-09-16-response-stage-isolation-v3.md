# Response stage isolation V3

Source `6b8b4e28537e8d6681492d1fefd28132652a3cb6`, launcher `f91d4eb`.
Artifacts: `test_results/response_stage_isolation_v3/`.

Eleven sequential compile-only stages ran. Compiled: `packing_control`,
`packing_selection`, `packing_guard`, `packing_first_dma`,
`packing_second_dma`, `exchange`, `receive`, `planes_to_wire`. Native aborts:
`packing`, `packing_gather`, and `composition`; each reports
`VectorLayout::join` rank `1 vs 2`. Runtime is JAX/jaxlib 0.10.2,
libtpu 0.0.42.1 on eight TPU v5 lite devices.

Guard and both DMA prefixes compile independently. Gather is the first newly
failing prefix, but the stack does not identify a precise source expression.
Next isolate per-plane gather from its DMA scratch and output. No execution,
correctness, latency, throughput, overlap, or end-to-end beam claim is made.

## Next diagnostic boundary

Source inspection shows that the gather prefix introduces several operations
together: a rank-one scratch load, dynamic position addition, explicit clipping
in `_take_clipped`, `take_along_axis`, a length mask, and a rank-one output store.
The successful DMA prefixes instead expose a rectangular scratch tile. This
confounds attributing the abort to gather itself.

Keep the existing successful second-DMA control and test these additional
observable variants in separate subprocesses:

1. Read and store each aligned scratch row without dynamic positions.
2. Expose dynamic positions before and after clipping as diagnostic output.
3. Gather each row with the same dynamic positions, without the length mask.
4. Apply the original mask to gathered rows.

If row-copy already fails, investigate row extraction/store layouts first. If
only clipping fails, compare equivalent bounded index expressions. If the raw
gather first fails, compare a rank-two gather using the same indices and data.
All variants must preserve runtime arguments and observable outputs; each is
diagnostic until physical compilation and execution have been checked.

Local artifact audit confirmed all eleven nested reports, expected source and
runtime, distinct device IDs 0..7, return codes, and HLO presence matching the
eight successful compilations. Publication remains pending after automatic
approval review rejected the previous publication command due to a usage limit.

Local follow-up variants have been added to the benchmark factory: `row_copy`,
`positions`, `clipped_positions`, and `unmasked_gather`. All four are registered
in the sequential bundle, making fifteen subprocess stages. Physical compilation
is pending. Position outputs do not depend on DMA data;
the compiler may eliminate those transfers. Therefore position probes isolate
index arithmetic and stores, not the complete DMA-to-gather layout context.
All legal shifts are 0..127, so position values are already in 0..254: clipping
does not change expected values and may itself be optimized away. Any physical
conclusion must inspect the retained IR as well as the compile status.

Local validation: five expected failures for missing registrations/coordinator
coverage, then 46 passing tests in 22.81 seconds across packing, stage tracing,
coordinator and launcher (`test_results/local_packing_followup_bundle.xml`).
The launcher still pins V3; no new source was published or remote run submitted.
