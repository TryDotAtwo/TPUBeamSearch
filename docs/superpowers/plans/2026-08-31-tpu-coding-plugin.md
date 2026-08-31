# TPU coding plugin implementation plan

> **For agentic workers:** Use independent task implementation/review and fresh-context forward testing. Follow the approved scope; do not dispatch remote TPU runs for plugin validation.

**Goal:** Publish a reusable, versioned Codex plugin that transfers verified TPU/JAX/Pallas lessons into future coding and experiment decisions.

**Architecture:** One `code-tpu` skill inside `plugins/tpu-coding`, with conditional technical references, a portable evidence ledger, application evals and a deterministic package checker. No MCP server, daemon or inference changes.

**Tech Stack:** Markdown, Codex plugin JSON, Python standard-library package tests; existing repository tests remain separate.

**Spec:** The approved routed-plugin proposal and `docs/research/2026-08-31-tpu-coding-research.md`, supplemented by `docs/research/2026-08-31-tpu-expert-followup.md`. User approved continuation and public publication.

## Global constraints

- Keep source in public TPUBeamSearch; normal scoped main push is authorized.
- Preserve BN/LN implementations and existing unrelated untracked artifacts.
- Distinguish documented behavior, measured results, hypotheses and superseded claims.
- Runtime API and generation limits are scoped, not timeless TPU constants.
- Plugin package must work after copying out of this repository. Link project evidence through pinned public URLs.
- No new Kaggle job, automatic updater, MCP service or third-party checkpoint redistribution.
- Publishing project material does not grant authority for unrelated future projects, credentials or remote mutations.
- Personal registration, if feasible, must preserve other marketplace entries. Packaging alone must not be called installation.

## Tasks and verification

### Task 1: Baseline application evaluation

- [x] Write realistic prompts with raw artifacts before skill instructions.
- [x] Run an independent agent without the new skill; save actual answers.
- [x] Score specific unsupported conclusions, missing controls and wrong arithmetic; use observed failures to shape references.
- [x] Classify as technical reference/technique, not a discipline/wording intervention (five-replicate wording microtests are not applicable).

### Task 2: Package and technical references

- [x] Scaffold `plugins/tpu-coding/.codex-plugin/plugin.json` with the prescribed creator; replace scaffold metadata with actual project metadata.
- [x] Write `skills/code-tpu/SKILL.md`: correct lowercase name, valid frontmatter, discriminating third-person `Use when` description, concise overview, source routes and conditional reference table.
- [x] Write references for hardware/runtime, layout/pipelines, numerical validation, benchmarking/scaling, and maintenance. Include one directly useful complete Python boundary-control example.
- [x] Include scoped project case studies, source dates and evidence IDs; do not copy external manuals or promote audit hypotheses into causes.
- [x] Add release README and evidence record format for maintenance after new measurements. Installation must not depend on repository-relative escaping links.
- [x] Verify generated manifests and skill frontmatter using official local validators.

### Task 3: Forward tests and package integrity

- [x] Run same application prompts in fresh context with the new skill; preserve actual responses and compare with baseline.
- [x] Independently review technical quality, spec compliance, source scope and installation portability; resolve material findings.
- [x] Add deterministic tests for local link resolution, evidence-source references and required metadata before implementing their checker; malformed fixtures must fail.
- [x] Record observed limitations, not claims of statistically proven agent improvement or TPU correctness/performance.
- [x] Remove unsupported prescriptions, duplicate narrative and unused supporting files; re-run affected tests after edits.

### Task 4: Publication and handoff

- [x] Run plugin validator, skill validator, package tests and the existing test suite; report skips or failures accurately.
- [x] Review scoped diff for secrets, rights issues, placeholders and unrelated changes.
- [x] Commit and push only plugin, its plan/evals and intentional repository links.
- [x] Register/install through the supported personal plugin workflow when feasible, verify effective state, and document the next-thread discovery boundary.
- [x] Link the published package and state what was and was not tested. No PR is needed: direct main publication is the explicitly approved workflow.

## Preflight and decisions

| Interface | Producer / consumer | Resolution |
|---|---|---|
| Evals | Task 1 raw scenarios / Task 3 identical forward scenarios | Evaluator receives no grading rubric or prior conclusions. |
| Reference paths | Task 2 routing / Task 3 integrity checker | Links resolve within the packaged plugin or use public pinned HTTPS evidence. |
| Evidence IDs | Task 2 ledger / Task 3 checker | Structural validity only; manual technical review establishes support. |
| Runtime state | Task 4 package / installed personal copy | Source is authoritative; installation is separately verified. |

Ruling: Work in the existing main checkout with exact file scopes, because the owner explicitly authorized scoped direct main publication. Do not move or clean external artifacts.

Ruling: Create one technical reference skill, not a suite of behavior-enforcing skills. A compact routing table replaces generic rules, pressure rhetoric and rationalization tables. No flowchart is needed.

Ruling: A project-local source package is required; a repo marketplace is not implied. Prefer the existing personal marketplace for local registration.

## Completion evidence

Completed 2026-08-31. Source plugin published in commit `82e7d44`; technical/spec re-review passed. Combined tests: 91 passed. Independent application evaluation: baseline 6/6 and with-skill 6/6 (no demonstrated success-rate improvement). See `test_results/tpu_plugin_eval_2026-08-31/report.md` for actual answers and limitations.

Personal plugin `tpu-coding@personal`, version `0.1.0+codex.20260831010242`, is installed and enabled. All 12 cached files matched source SHA256; other marketplace entries were preserved. No TPU run, inference implementation change or recurring automation was added.
