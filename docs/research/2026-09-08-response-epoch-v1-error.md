# Response epoch V1: lowering rejection

Private kernel `trydotatwo/tpu-beam-response-epoch-gate` reached ERROR.
Downloaded all available output into `test_results/beam_response_epoch_v1`.
Source: `8dc27ca763764c8485988c06210fddfc7c0f5854`; child returncode1.
Runtime: JAX/jaxlib0.10.2, libtpu0.0.42.1; eight unique TPU v5 lite devices.

Preparation failed at `prep.lower(*args)` on the first `empty` case.
The source-located exception identifies `beam_final_intervals.py:31`:
`jnp.sum(hits.astype(jnp.uint32), axis=1)`. Mosaic lowering rejects
unsigned integer reductions with `NotImplementedError`. This is a Python
lowering exception, not the earlier native VectorLayout abort.

No epochs executed; exact=false, epochs=[] in the only pending case.
There are no successful compilation/HLO or timing results to interpret.
Kaggle terminal status does not establish correctness for any of the33epochs.

Next fix: add a failing reduction-dtype regression, then use a bounded signed
reduction for the per-tile boolean count and convert the result back to the
uint32 storage ABI. Inspect other reductions in this same preparation path;
do not infer their TPU acceptance from interpreter success. Re-run full tests,
publish a new source SHA, then submit a successor only after terminal status.
The already running local scatter regression must finish before source edits.
