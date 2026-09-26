# Local validation snapshot isolation

The background full pytest run completed with **1145 passed, 16 skipped,
2 failed**, 1163 collected cases, in 1606.22 seconds. Raw local outputs are
`test_results/local_tpu_architecture_postfix_2026_09_26.{xml,stdout.log,stderr.log}`.
This is a mixed-source run, not acceptance of a specific commit.

Both failures were in `test_beam_state_width_bridge_execution.py`:

- `test_width_gate_oracle_covers_all_eight_distinct_devices`: loaded test
  expected five values, while `make_inputs()` returned six.
- `test_width_gate_validator_rejects_high_byte_corruption`: loaded test
  expected two values, while `expected()` returned four.

During that process, the bridge gate was extended to include compact-frontier
scatter. The corresponding tests and module were updated together, but pytest
had already loaded the older tests. The traceback displayed current source
lines alongside the older unpacking exception. This establishes why a live
editable checkout cannot be used as an immutable validation snapshot.

The current three bridge-execution tests pass (3.36 seconds). No additional
production repair was made for these two failures. Separate checks also passed:
36 publication/history tests, one failure/retry history test, and 40 S4
ready/commit/collector tests after the sticky-fatal admission repair.

Next full validation is pinned to `4cbdf3842356564c6cc72741b68d60e81630a31d`
in a dedicated managed checkout. Do not edit that checkout during execution.
Record the resulting JUnit separately. A skipped source oracle remains skipped,
and local interpretation does not establish physical TPU or CUDA parity.

The first managed-checkout attempt failed on a long archived profile filename.
Git `core.longpaths=true` was enabled locally for TPUBeamSearch, not globally.
No user data or existing worktree was deleted.

The production-packing Kaggle gate remains queued. No replacement session was
launched. No new throughput or overlap claim follows from these local checks.
