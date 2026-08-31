# Execution-boundary A/B v1: queued

On2026-08-31, the prepared private bundle was submitted **once**. Kaggle
acknowledged version1; live status was **QUEUED at17:14:09 UTC**.
[Notebook](https://www.kaggle.com/code/trydotatwo/tpu-execution-boundary-ab).
See the exact [submission receipt](submission.json).

Source: `45062324d368f4849adb6d572d21d54f75854d79`.
Launcher: `51c8b3a512e650df83939d322c78bd715cfd8221`.
Both were public before submission; remote main was verified. No inference
code or numerical gate changed during this launch turn. The launcher contract
test passed freshly; the prior complete282-test run belongs to the unchanged
published implementation, not a newly repeated full regression here.

## Occupancy and access checks

The previous403 failure was transient. Authenticated CLI listing succeeded
again without changing proxy, VPN or credentials. Some status/list calls still
failed with TLS EOF or proxy disconnection; bounded read-only retries resolved
the outstanding statuses before submission.

Listing sorted by most recent run exposed20 existing project TPU notebooks.
All20 had terminal statuses:19 COMPLETE and the old block-diagnostic ERROR.
Two additional recent owned jobs were also terminal. No other notebook was
restarted or changed. The newly prepared slug was absent before submission.
The list endpoint's unset/default `enable_tpu=False` fields were **not** used
as proof that old notebooks ran on CPU; actual statuses were queried.

## Server-side verification

After submission, source and metadata were pulled back from Kaggle into
[`server_bootstrap/`](server_bootstrap/kernel-metadata.json).

- Private:true, TPU:true, GPU:false, attached dataset
  `artgor/cube555-tpu-artifacts`.
- Downloaded bootstrap has the exact pinned source SHA and matches the
  published launcher text after newline normalization. AST validation passed.
- Requested accelerator was `v3-8`; **server metadata selects `TpuV5E8`**.
  This is scheduling metadata, not an observed runtime device inventory.
  Runtime generation, active devices, versions and hashes remain pending.

## Next checkpoint

The existing15-minute heartbeat `check-tpu-execution-boundary` now records that
v1 is already submitted. Do not push again while QUEUED/RUNNING. A transient
status-request failure does not imply the session disappeared.

After terminal completion, download output and the complete Kaggle log into
this run's directory without overwriting the submission artifacts. Analyze
`execution_boundary/stream1_execution_boundary.json` according to the
[frozen protocol](../../docs/research/2026-08-31-execution-boundary-bundle.md):
all declared cases and rejected compiles, direct witnesses, exact dual16K Q
gate, separate actual32K confirmation, comparable timing and diagnostic
profiles. This launch report contains **no TPU correctness or speed result**.
