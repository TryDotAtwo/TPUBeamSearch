# Maintaining the TPU coding knowledge

Update when a completed experiment, reproducible failure, source-code audit or
verified upstream change alters a decision. This is an evidence-triggered
maintenance procedure, not an unattended schedule or authority for new runs.

## Evidence record

[evidence.json](evidence.json) has `schema_version: 1` and an `entries` list.
Each entry contains:

| Field | Meaning |
|---|---|
| `id` | Stable, unique lookup key; do not reuse it for a different claim. |
| `status` | `documented`, `measured`, `hypothesis`, or `superseded`. |
| `claim` | Narrow claim, with old incorrect interpretation explicitly labeled. |
| `scope` | Hardware/runtime, architecture/shapes/input and timing boundaries; explicitly record unknowns. |
| `source_urls` | Primary documentation or pinned public code/results; expert summaries identify advice. |
| `checked_on` | ISO date of verification, not a promise of timeless validity. |
| `recheck_when` | Observable trigger: changed runtime, shape, generation, consumer or conflicting result. |
| `regression_case` | Test/control that could reject the claim or catch its misapplication. |

`documented` includes verified source/code facts and mathematical contracts,
not a new execution measurement. `measured` records what the harness actually
measured, not an inferred mechanism. A source-level rounding difference may
be documented while its contribution to TPU drift remains a hypothesis.
`superseded` preserves the old interpretation and links the correction; it
does not erase or silently relabel raw JSON.

## Change procedure

1. Preserve source SHA, result JSON, useful logs, runtime/device inventory and
   input provenance. Check credentials, unrelated data and third-party rights
   before making any artifact public.
2. Add or amend the narrow evidence record. State what changed and what remains
   unknown; an expert's agreement alone cannot upgrade a hypothesis to measured.
3. Update the routed reference only if the finding changes a reusable decision.
   Keep experiment-specific tiles/numbers in case studies, not main defaults.
4. Run the affected application scenario and a fresh-context evaluation. Supply
   raw artifacts, not the intended diagnosis; preserve prompts and actual answers.
   Use a no-skill baseline for a new claimed behavior improvement. Record
   successful baselines honestly; a single pass is not statistical evidence.
5. Validate the manifest/skill with the available Codex creator validators and
   run this package's integrity checks and tests. Re-run runnable examples after
   API changes. Interpretation and local CPU checks remain separately labeled.
6. Bump strict semantic version for content changes; the local creator's
   cachebuster adds build metadata when refreshing an installed plugin. Publish
   a scoped commit under the current project's authorization and reinstall the
   updated personal copy. Verify actual installed version; existing threads may
   require a fresh task to discover changed skills.

## Regression coverage

Retain scenarios for JAX-only boundary controls; Dense bias rounding; LN width
130 stored as 256; statistics dtype/epsilon; inverse masks and minimizing global
top-K; ties and insufficient valid candidates; real versus categorical inputs;
captured/runtime parameters; VMEM rejection; matched-tile versus tuned selection;
weak/strong/search scaling; stale `pallas_call`/`pl.kernel` examples.

Run from the plugin root:

```text
python scripts/check_package.py .
python -m unittest discover -s tests -v
```

The checker validates packaging and evidence structure, not external URL
availability, scientific truth, hardware compilation or inference quality.
Use the [release README](../../../README.md) for public source and installation.
