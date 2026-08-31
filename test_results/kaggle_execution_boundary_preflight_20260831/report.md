# Execution-boundary bundle: prepared, not submitted

Historical preflight snapshot. The later
[successful v1 submission](../kaggle_execution_boundary_v1/launch.md)
supersedes the access blocker and not-submitted state below.

Date:2026-08-31. **No new Kaggle kernel/session/version was created.**

Public source commit:
`45062324d368f4849adb6d572d21d54f75854d79`, verified against
`TryDotAtwo/TPUBeamSearch` main immediately after the scoped push.

Prepared private package: `kaggle_execution_boundary/`.
Intended slug: `trydotatwo/tpu-execution-boundary-ab`.
The bootstrap fetches that exact public SHA and runs
`benchmarks.stream1_execution_boundary`, with JAX/jaxlib0.10.2 and
libtpu0.0.42.1. It preserves console output in
`/kaggle/working/execution_boundary/benchmark.log`; result JSON is
`stream1_execution_boundary.json` in the same directory.

Protocol: [execution-boundary bundle](../../docs/research/2026-08-31-execution-boundary-bundle.md).

## Verified locally

- Source-only suite:281 passed in102.81s.
- Launcher contract test first failed because its package did not exist,
  then passed after implementation.
- Fresh complete suite including launcher: **282 passed in105.86s**.
- Python compilation, source/module CLI help with `src` on `PYTHONPATH`,
  private metadata, full source SHA, dependencies and log-preserving bootstrap
  all checked. Source changes and scoped staged diff were reviewed inline.
- No BN/default changes; unrelated local artifacts remain untouched.
- CPU/interpreter/JAXPR checks do not establish TPU compilation or speedup.

## Access blocker

At approximately16:33–16:46 UTC, ordinary CLI status queries for the existing
project TPU notebooks, including the completed arithmetic follow-up, returned
HTTP403. `kaggle kernels list --mine` also returned403, so this is not evidence
that each individual notebook is absent or that the TPU slot is free.

The raw status endpoint response was generic HTML Forbidden, not a structured
kernel status. The CLI rewrites401/403 into a generic kernels.get permission
message; that message alone does not establish a wrong slug or revoked key.

Bounded read-only checks reproduced403 outside the shell sandbox and through
the legacy status endpoint. The configured proxy remained unchanged; one
isolated direct read also returned403. Existing legacy credentials were used
without displaying, modifying or publishing them. No system proxy/VPN settings
were changed. Browser fallback timed out and its connection subsequently became
unavailable; it provided no authoritative notebook status.

The exact upstream/network/authentication cause is unresolved. GitHub publication
succeeded after its separate Windows sandbox Schannel credential issue was
handled by the normal approved shell escalation.

## Resume safely

1. Recheck authenticated CLI listing/status with the existing configuration.
   If denied, record that access is still unavailable; do not submit blindly.
2. From a successful owned-kernel list, check actual existing project TPU
   notebooks and any active owned TPU session. Prepared-but-never-submitted
   slugs need not exist. Never restart QUEUED/RUNNING sessions.
3. Only after confirming a free slot, submit the published private package
   once. Verify the response, actual version and live status. If submission
   outcome is ambiguous, reconcile server state before any retry.
4. Record requested accelerator separately from actual device inventory; the
   previous requested v3-8 jobs actually reported TPU v5 lite.
5. Monitor terminal output, download to a new versioned directory, analyze all
   cases against the frozen gate and publish scoped evidence. Retire the
   experiment's monitor after its final report.

The recurring follow-up may retry access and perform this already-authorized
launch once access and occupancy are verified. It must work without subagents.
