# TPUBeamSearch

Pallas/JAX port and TPU validation of the MultiGPUBeamSearch pipeline.

The implementation is developed here first. Kaggle launchers clone an exact
Git commit; published run reports preserve source pins. New result JSON should
also include the SHA explicitly (some historical artifacts do not).

## Public collaboration

TPUBeamSearch is a public open-source project. The maintainer explicitly
authorizes publishing its code, benchmark results, useful technical logs,
research and the `tpu-coding` plugin in this repository, and sharing
relevant project material with the selected project experts. Public technical
material does not need a separate confidentiality approval for each discussion.

This authorization covers this project, not credentials, access tokens,
unrelated private data or materials we do not have redistribution rights for.
Existing third-party licenses and tool-level safety checks still apply.
Publish scoped changes with sources and preserve the distinction between
measurements, expert recommendations and hypotheses.

See the [TPU coding research](docs/research/2026-08-31-tpu-coding-research.md)
and [detailed expert follow-up](docs/research/2026-08-31-tpu-expert-followup.md).

## TPU coding plugin

[TPU Coding](plugins/tpu-coding/README.md) packages the audited JAX/Pallas
knowledge as one routed Codex skill. It includes runtime/layout guidance,
numerical and ranking controls, benchmarking methodology, scoped BN/LN case
studies and a versioned evidence-maintenance procedure. It does not change
inference defaults or launch experiments automatically.

## Stream1 inference contract

`Stream1Architecture` is the immutable, compile-time model description.
`Stream1Weights` is a JAX PyTree whose arrays remain dynamic arguments to the
compiled executable. `stream1_weights_from_pytorch_state_dict` folds inference
BatchNorm parameters and converts PyTorch `[output, input]` matrices to the
Pallas `[input, output]` layout.

```python
architecture = Stream1Architecture.from_pytorch_state_dict(
    state_dict,
    STATE_LEN=120,
    STATE_STORAGE_LEN=128,
    NUM_CLASSES=120,
)
weights = stream1_weights_from_pytorch_state_dict(state_dict, architecture)
infer = make_jitted_stream1_inference(architecture, backend="pallas")
logits = infer(states, weights)  # [batch, MOVE_COUNT]
```

The current Pallas path fuses virtual-one-hot input and the first dense hidden
layer. Residual matrices and the output head use reusable aligned dense Pallas
kernels. This is the correctness baseline for later residual fusion experiments.
