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
