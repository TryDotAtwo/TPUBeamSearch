# TPU Coding 0.1.0 release evaluation

Date: 2026-08-31. This evaluates a technical-reference plugin, not TPU inference
quality or performance. No remote TPU job was launched and neither BN nor LN
implementation was changed.

## Application evaluation

The [six raw scenarios](../../docs/research/tpu-plugin-eval-scenarios.md) were
written before the skill. One fresh-context baseline agent received only those
fixtures and access to primary documentation. A different fresh-context agent
received the same fixtures plus the completed routed skill. Neither received
the intended answers, scoring notes or the other's response.

Actual answers: [baseline](baseline.md), [with skill](with_skill.md).

| Scenario | Core observable criterion | Baseline | With skill |
|---|---|---|---|
| A: depth drift | Detect JAX-boundary confound, same-suffix control, no aggregate-equality or rounding-cause overclaim | Pass | Pass |
| B: promotion | Use minimizing flattened selection, masks/ties/valid count, reachable inputs and replay | Pass | Pass |
| C: padding | LN population stays logical; layout exceptions and interpreter limitations remain scoped | Pass | Pass |
| D: timing | Keep faster current full-model JAX; separate tile/fusion attribution and weak/search scaling | Pass | Pass |
| E: pipeline | Inspect actual runtime; ANY is unconstrained; buffering is not proof of overlap | Pass | Pass |
| F: evidence | Captured/runtime A/B on both implementations, rejection/invalid separation, explicit unknowns | Pass | Pass |

The baseline already answered all six correctly. **This does not demonstrate
an improvement in agent success rate.** The observed benefit is packaging and
retrieval of scoped project evidence, not a measured behavioral speedup or
accuracy gain. The historical failures in the published audit motivate the
reference; they were not fabricated as failures of this baseline. One sample
per condition cannot establish reliability or variance. No five-replicate
wording intervention was attempted: this release is a technical reference.

The with-skill response additionally used the evidence schema and explicitly
crossed JAX/Pallas Dense and LN, but the evaluation does not attribute this
difference causally to the plugin from one pair of samples.

## Independent review

A separate technical reviewer read the package, approved plan, audit and expert
follow-up and checked primary JAX/Cloud sources. Technical accuracy and portable
links passed. The reviewer requested evidence IDs/source records for the retained
BN prefix-fusion and residual near-tie observations; these were added as
`M-BN-FUSION` and `M-BN-RESIDUAL` with the original reports and limitations.
The scoped re-review marked that fix addressed with no new material finding.

Review did not test hardware execution or claim that the existing LN comparison
is now repaired. The JAX-only controls and task-aware gates remain requirements
for a future inference experiment.

## Mechanical and local checks

- Codex plugin manifest validator: passed.
- Codex skill validator: passed.
- Offline package checker: passed; relative inline links and evidence structure
  resolve independently of the repository. It is not a general Markdown parser,
  secret scanner, network availability checker or scientific validator.
- Standard-library package tests: **20 passed**. Tests include missing/escaping
  links, metadata, malformed/non-object/null JSON, invalid status types, malformed
  HTTPS URLs, boolean schema versions and actual CLI exit codes.
- Combined `python -m pytest -q tests plugins/tpu-coding/tests`: **91 passed**
  (71 existing repository tests plus 20 package tests), 34.46 seconds locally.
- The reference's extracted three-block JAX boundary-control example ran on CPU
  JAX/jaxlib 0.10.1; boundary, whole-model and zero-replacement comparisons each
  printed `0.0 True`. Nonzero drift is not required for this illustrative control.

Test-development record: the initial absent checker produced a loader error,
not a behavioral RED. Later review found real false accepts (`[]`/`null` JSON
roots, boolean schema) and exceptions (list-valued status and malformed URL).
New assertion tests reproduced these before fixes, then passed. This distinction
is retained instead of describing the initial import error as adequate TDD proof.

## Release scope

Personal installation was completed through `codex plugin add tpu-coding@personal`.
Installed version: `0.1.0+codex.20260831010242`. `codex plugin list` confirmed
`installed: true`, `enabled: true`; all 12 cached package files matched the
repository source by SHA256 and the cached package checker passed. The other
seven marketplace entries were preserved. This verifies registration and cache,
not automatic selection inside an already-running task.

Public source is authoritative. Personal installation is a separate local copy;
new tasks discover the installed skill. No recurring updater, remote execution permission, credentials or
third-party checkpoint is included. Maintenance is triggered by verified new
evidence, with source/version updates and re-evaluation rather than silent
changes to historical measurements.
