# Runtime and hardware

Checked: 2026-08-31. Use when selecting APIs, interpreting allocation failures,
or comparing hardware. These are versioned reference facts, not a requirement
to provision hardware for every question.

## Identify the executing target

Record Python, `jax.__version__`, `jaxlib.__version__`, installed `libtpu`
version when available, relevant compiler flags, `jax.default_backend()`,
`jax.process_index()`, `jax.process_count()`, `jax.device_count()` and
`jax.local_device_count()`. For each `jax.devices()` entry preserve `id`,
`device_kind`, `process_index`, and exposed coordinates/core information.
Record mesh shape and array sharding separately. A process, JAX device,
physical TensorCore, chip and mesh axis are different units; establish their
mapping for the actual slice. See [JAX multi-controller documentation](https://docs.jax.dev/en/latest/multi_process.html).

A requested accelerator name, generic warning, or `str(device)` alone is not
sufficient generation evidence. Historical project reports contain conflicting
v3/v5e labels: [case studies](case-studies.md) preserve that uncertainty.

## v3 example: units and conflicting peaks

[Cloud TPU v3](https://docs.cloud.google.com/tpu/docs/v3) documents two
TensorCores per chip, two MXUs per TensorCore, and 32 GiB HBM per chip: 16 GiB
per TensorCore. The [JAX hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
lists per-TensorCore VMEM 16 MiB, SMEM 16 KiB, and no SparseCore for v3.
Its displayed HBM figure is 17 decimal GB, consistent with 16 GiB after
display rounding; do not interpret it as 17 GiB.

The same sources disagree numerically on peak compute/bandwidth: JAX lists
70 BF16 TFLOP/s and 412.5 GB/s per TensorCore; Cloud lists 123 BF16 TFLOP/s
and 900 GB/s per chip. Doubling JAX's figures does not reproduce Cloud's.
Preserve source and configuration when choosing a utilization denominator;
do not silently reconcile these values. Other generations have different
memory and compute resources.

## Match the installed API

Project TPU reports used JAX 0.10.2; the audit's local CPU runtime used 0.10.1.
Neither pins what `latest` documentation describes. Check installed signatures
before adapting `pl.kernel`, `pl.pallas_call`, `CompilerParams`, scratch or
semaphore examples. In the [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html),
0.10.1 removes `pallas_call` checkify support; 0.10.2 defaults TPU
`needs_layout_passes` to true. Mosaic-GPU migration advice is not a TPU-wide
deprecation. Match release and backend, including unreleased sections.

Plain `interpret=True` is HLO interpretation. Supported versions also offer
[`pltpu.InterpretParams`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)
for TPU memory/DMA/synchronization simulation and optional race detection.
Enabled detection is not exhaustive proof; neither interpreter establishes
real TPU compilation, synchronization safety or performance. For target
measurement see [benchmarking and scaling](benchmarking-scaling.md).
