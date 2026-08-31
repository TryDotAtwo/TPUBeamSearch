# LayerNorm arithmetic A/B v1 launch

Date: 2026-08-31.

- Private kernel: `trydotatwo/tpu-layernorm-arithmetic-ab`, version 1.
- Requested accelerator: `v3-8` through the Kaggle CLI. Actual device kind must
  be read from runtime inventory; the request is not hardware verification.
- Benchmark source: `2e9602829b8e4fa8498b64461f64c556e77ad4f4`.
- Published launcher: `e1ef75d8b099b6799b446130737032f759c6a632`.
- Dependencies pinned: JAX/jaxlib0.10.2, libtpu0.0.42.1.
- Local verification: 142 pytest cases passed, Python compilation and Git
  whitespace checks passed. Independent review found no remaining prelaunch
  blocker. No TPU performance/correctness result is claimed yet.
- Latest launch-time status: `KernelWorkerStatus.RUNNING`.
- No other checked LayerNorm session was QUEUED/RUNNING. The historical block
  diagnostic was ERROR, superseded by completed comprehensive/depth runs;
  it was not restarted.
- Recurring monitor was **not created**: the automation approval check rejected
  persistent checks with future GitHub publication as requiring fresh explicit
  authorization. The one-off experiment remains running independently.

The [experiment contract](../../docs/research/2026-08-31-layernorm-arithmetic-ab.md)
describes 30 block candidates and eight full-model candidates per corpus,
controls, numerical/ranking metrics, eligible timing scope and bounded promotion.

Expected kernel output: `arithmetic_ab/stream1_layernorm_arithmetic.json`,
`arithmetic_ab/benchmark.log`, HLO files and optional profiles. Download terminal
output into this directory without touching unrelated existing artifacts. Inspect
case errors even if Kaggle itself reports COMPLETE. Preserve raw results before
writing the measured report; this file is only a launch record.

## Monitoring authorization follow-up

On 2026-08-31 the maintainer explicitly authorized project publication and
automations. The app then created `check-tpu-arithmetic-a-b` with status `ACTIVE`,
checking this experiment every ten minutes. It may publish scoped results to
the existing public `main`, diagnose/fix/retry this failed experiment, and must
remove itself after the final report. It must not restart QUEUED/RUNNING work,
change the BN path, publish secrets, or touch unrelated artifacts. The earlier
creation rejection above is historical; monitoring is now enabled.
