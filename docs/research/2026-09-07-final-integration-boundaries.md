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

## Remote result acceptance contract

The pending two-process coordinator must require exactly these case identities:

- Scatter: count0_valid, count1_valid, count127_valid, count128_valid,
  count129_valid, count129_count_overflow, count129_target_overflow.
- Integrated: the five valid cases above; invalid upstream materialization is
  outside this diagnostic's contract and must not be represented as safe commit.

Reject duplicates, missing/extra cases, wrong mode, nonzero process returncode,
pending/compiling status and malformed metrics even if a nested exact flag is
true. Require eight mismatch entries equal to zero, eight expected error-count
entries (one for overflow fixtures, zero otherwise), and eight SHA values equal
to the expected frontier SHA. Check source SHA and eight physical TPU devices
against the pinned launch. A fresh destination avoids stale nested results.

Save coordinator JSON before each child and after return, retain native signal
returncode and partial reports, and continue the other child after a crash.
This contract is additional to the runner's own final exact flag; it must be
tested with false-positive fixtures before launch. Physical output byte
equality remains the gate, not CPU pass count or kernel COMPLETE status.

The coordinator must derive expected input/output hashes and error counts from
the local pinned fixture, not trust the nested report's expected fields. A
self-consistent but wrong expected/output hash pair must fail acceptance.
Include that mutation in the false-positive tests. The coordinator may import
the NumPy-only fixture module; it must not import JAX or initialize TPU before
spawning children. Device IDs must be unique, not merely eight list entries.
Use strict integer checks for counters so JSON booleans cannot pass as zero or
one through Python equality. Runtime version fields must be nonempty and match
between both child reports before declaring the bundle accepted.

## Launcher migration checklist

The existing `kaggle_beam_final_gate/run.py` still launches V7's
`benchmarks.beam_final_bundle` and creates its output directory beforehand.
For the scatter bundle, remove that precreation: the new coordinator owns the
fresh-directory check and intentionally rejects an existing destination.
Switch the module to `benchmarks.beam_final_scatter_bundle`, pass
`--source-sha` equal to the pinned checkout SHA, and use a distinct
`/kaggle/working/beam_final_scatter` output. Keep the runtime pins and private
single-session policy. Test the launch command and directory ownership before
submission; merely changing the module would fail before either TPU probe.
