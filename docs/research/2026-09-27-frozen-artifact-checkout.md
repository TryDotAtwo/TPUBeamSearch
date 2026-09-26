# Frozen artifact checkout diagnosis

The isolated full suite at `4cbdf3842356564c6cc72741b68d60e81630a31d`
finished with **1172 passed, 16 skipped, 2 failed** in 1295.40 seconds.
It does not validate later S4/S5/runner commits.

Failures:

- `test_artgor_snapshot_is_the_frozen_script_version`
- `test_generated_notebook_records_its_frozen_source_and_builder_hash`

Git `core.autocrlf=true` converted the frozen `jax_model.py` and generated
exact notebook from index LF to working-tree CRLF. Converting only CRLF to
LF reproduced their existing expected SHA-256 values exactly. The snapshot
`jax_beam_spmd_v_only.py`, in contrast, already matches its manifest with
CRLF bytes; it was not changed or normalized by this fix.

The scoped fix pins `text eol=lf` for the two failing paths. No model,
notebook, manifest, baseline, or expected digest was changed.

Validation: the two test modules pass (10 tests). Git's actual checkout
filter (`git -c core.autocrlf=true cat-file --filters HEAD:<path>`) produces
the expected SHA-256 for both paths with the new attributes. `git diff
--check` passes. A new complete-suite run remains pending; the earlier
full-suite result remains red and is not relabeled as passing.

Kaggle production packing remains QUEUED during this diagnosis; no duplicate
TPU job was submitted and no accelerator correctness/speed claim follows.

## Existing-checkout follow-up

The isolated `944478f` run finished with **1188 passed, 16 skipped, 2 failed**
in 2508.86 seconds. The same two hash checks failed: switching to the commit
containing attributes did not rewrite existing CRLF working-tree files.
`git ls-files --eol` confirmed `i/lf w/crlf attr/text eol=lf` for both paths.
Even `checkout-index --force` retained their bytes in this checkout.

Explicitly normalizing only these two known-clean files to LF made both test
modules pass: **10 passed in 2.33 seconds**. Expected digests and source content
were unchanged. After switching the verification checkout to `2044dac`,
`git diff --stat`, `git diff --numstat`, and the whitespace-insensitive diff
were empty. A complete run of that immutable revision was started separately;
its result is pending and does not cover subsequent response-plan edits.

That complete `2044dacf8abdc9b0f2246a01b64b0ef432f12cd8` run has now
finished: **1255 passed, 16 skipped in3560.81 seconds**. The final stdout and
verification-worktree JUnit report identify this snapshot. This resolves the
full-suite checkout-byte failures for that revision; it does not validate the
later streaming coverage, response/history consumers or materialization round.
