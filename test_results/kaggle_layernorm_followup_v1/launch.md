# Arithmetic follow-up v1 launch

2026-08-31, approximately06:22 Europe/Moscow (03:22 UTC).

- Private kernel: [trydotatwo/tpu-layernorm-arithmetic-followup](https://www.kaggle.com/code/trydotatwo/tpu-layernorm-arithmetic-followup).
- Kaggle accepted **version1**; status check at06:22:37 MSK returned **RUNNING**.
- Source commit: `d58cf9fd8e86ec145c6bbc4f6c7f5aff489d6e21`.
- Launcher commit: `d87fa2b16d5bc3489d914939db0ce4ba7766b397`.
- Both commits were pushed to public `TryDotAtwo/TPUBeamSearch`, main, before Kaggle submission.
- Submission: `kaggle kernels push -p kaggle_layernorm_followup --accelerator v3-8`.
- Bootstrap pins JAX/jaxlib0.10.2 and libtpu0.0.42.1, verifies detached source SHA, then runs `benchmarks.stream1_layernorm_followup`.
- Output target: `/kaggle/working/arithmetic_followup/stream1_layernorm_followup.json`, plus full subprocess log, StableHLO/compiled HLO and diagnostic profiles.

The accelerator request is **not verified runtime hardware**. Arithmetic v1
received TPU v5 lite/v5e despite the same request. No log was yet available
at the initial follow-up log check, so device/runtime/actual benchmark execution
remain unconfirmed. RUNNING alone is not a numerical or speed result.

## Local verification

`python -m pytest -q`: **252 passed in55.70s**, exit0. Real CPU compiled/interpreter
tests cover kernels, rounding/padding, runtime controls, paired/queued timing,
negative-quality profiling, failed-group salvage and strict cross-corpus gates.
`compileall` validated launcher syntax; JSON metadata/private TPU/source pin
checks passed. Staged diffs passed `git diff --cached --check`.

Independent review identified sequential singleton microtiming and uncaught
group timing failures. Regression tests first failed, both issues were fixed,
and scoped re-review found no remaining blocking findings. Production BN/LN
defaults and unrelated untracked artifacts were not changed or staged.

## Session preflight and monitoring

Before submission, all13 existing TPU-named runs in the inspected account list
were terminal (12 COMPLETE,1 historical block-diagnostic ERROR). No existing
job was restarted or stopped. Only this new job was submitted.

Heartbeat **`check-tpu-arithmetic-follow-up`** was created ACTIVE for this task,
checking every10minutes. It preserves QUEUED/RUNNING jobs, downloads terminal
outputs/logs, compares provenance against v1, analyzes the frozen protocol,
publishes scoped project artifacts and deletes itself after the final report.
Errors require reproducible diagnosis, a failing test, checked GitHub fix and
a new source pin before restarting only this kernel.

Protocol: [arithmetic follow-up bundle](../../docs/research/2026-08-31-arithmetic-followup-bundle.md).
No TPU speedup, exact-Q result,32K confirmation or scaling result is claimed yet.
