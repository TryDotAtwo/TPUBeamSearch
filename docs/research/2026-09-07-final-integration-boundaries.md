# Next final integration boundary

V7 validates materialization, exchange and coverage separately. It does not
exercise `pallas_scatter_final_responses` on physical TPU. Current scatter
still uses two-dimensional uint8 HBM views, one-row dynamic DMA and one-row
VMEM staging. Materialization needed a record-axis layout to pass the analogous
V2 rejection. This is a source-based risk, not a measured scatter compile error.

The next smallest integrated physical gate should run materialization then
response unpack/scatter on identical CUDA-fixture bytes, preserving unrelated
frontier rows and zeroing the response-index tail. Include target0 and a
non-eight-aligned target, boundary counts127/128/129, zero count, and rejected
count/target overflow. Compile scatter separately too to localize rejection.
Do not infer scatter acceptance from V7.

Only after this local integration passes should the gate route requests and
responses between ranks. `pallas_materialize_final_snapshots` retains both
receive and validation summaries, and return-rank metadata alongside wire.
Neither summary may be dropped. Remote ranks require the appropriate target
capacity contract; a single local capacity is not automatically every
destination's capacity.

Coverage agreement is a common-error decision, not a DMA drain. Scatter output
must remain private until all transfer/coverage/history errors are agreed and
device work completes. No caller may reuse aliased scratch or publish history
because a copy was merely scheduled. Host history commit must follow successful
completion, and failure must leave the previously published frontier intact.

Production caller, shared scratch aliases and full ownership/lifetime protocol
remain outstanding. This document is a test sequence, not implementation or
physical evidence. No performance claim.
