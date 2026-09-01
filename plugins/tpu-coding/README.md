# TPU Coding for Codex

Version 0.1.1. A maintained Python/JAX/Pallas technical skill built from primary
TPU documentation, TPUBeamSearch experiments and project-expert review.

Start with [code-tpu](skills/code-tpu/SKILL.md). It routes to hardware/runtime,
layout/pipelining, numerical validation, benchmarking/scaling and maintenance.
The [evidence registry](skills/code-tpu/references/evidence.json) records source,
date, scope, status and a regression case for each retained project claim.

This is a skill-only plugin: no MCP server, credentials, daemon, automatic TPU
jobs or inference implementation. It does not require other personal plugins.
English technical references support use across projects; user replies follow
the current conversation language.

## Use and installation

Install the `tpu-coding` package through your configured Codex marketplace.
For a personal local marketplace whose root is your home directory, place this
folder at `~/plugins/tpu-coding`, register only its entry using the available
Codex plugin-creator workflow, then run:

```text
codex plugin add tpu-coding@personal
codex plugin list --marketplace personal --json
```

The first command requires an existing marketplace entry; copying files alone
does not install a plugin. Use a fresh task after installation to discover the
skill. Invoke `tpu-coding:code-tpu`, or ask for a TPU/JAX/Pallas review so ordinary
skill selection can discover it. Keep public source authoritative and refresh
the installed copy after a versioned change.

Examples: inspect a Pallas VMEM failure; design a fair full-model JAX/Pallas
A/B; check padding and LayerNorm semantics; update evidence after a completed
TPU experiment. The plugin does not assume Pallas must beat JAX.

## Checks and updates

From this folder, with Python 3.11 or newer:

```text
python scripts/check_package.py .
python -m unittest discover -s tests -v
```

These standalone checks cover links, evidence metadata and negative fixtures.
They supplement the Codex plugin/skill validators; they do not prove TPU
performance, numerical correctness or that upstream pages remain current.
Follow [maintenance](skills/code-tpu/references/maintenance.md) after new evidence.

The original audit, expert advice, evaluation prompts and release evaluation
are maintained in the [public repository](https://github.com/TryDotAtwo/TPUBeamSearch).
Do not redistribute credentials or third-party checkpoints with this package.
