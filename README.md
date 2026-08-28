# TPUBeamSearch

Pallas/JAX port and TPU validation of the MultiGPUBeamSearch pipeline.

The implementation is developed here first. Kaggle launchers clone an exact
Git commit and record its SHA in every result artifact.

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
