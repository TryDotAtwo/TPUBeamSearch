# Artgor Pallas same-suffix v2

The causal diagnostic completed on eight Kaggle TPU cores for legal seed 42 and
stress seed 43 at 256 states/device.

- `input_stack` is already inexact: 21,982/24,861 hidden BF16 mismatches and
  9,624/27,957 final-Q mismatches through the shared JAX suffix.
- Every isolated residual block is inexact on the supplied JAX hidden input.
- The isolated Pallas head is bitwise exact on both corpora.
- The zero-replacement same-suffix control is exact at even residual depths and
  depth 10. At the input and odd depths it differs from monolithic JAX, proving
  that these control differences are compilation-boundary drift, not Pallas
  error. Candidate-versus-control remains the causal metric.

The first causal mismatch is therefore inside the input stack. The follow-up
splits it into embedding, input Dense, and input LayerNorm+ReLU, with each
candidate and reference boundary fed through the same suffix executable.
