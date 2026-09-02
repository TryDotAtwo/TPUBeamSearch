# Unmasked all-Pallas diagnostic v7

Private Kaggle kernel `trydotatwo/tpu-artgor-all-pallas-exact-diagnostic` v7
completed from source `1f43f79414e54fdc3cce14a92b154e4d3048d0bd`
on eight TPUv5lite devices.

Removing the aligned-width predicate/select path did not change a single
measured error: all six `input.layernorm_relu` mismatch counts and error values
are identical to v6 (21,165--25,312 BF16 elements, max abs
0.0078125--0.015625). The predicate hypothesis is therefore rejected.

The standalone `fp32_variance` probe remains hash exact on the same operator
and inputs. V8 reuses that exact TPU-proven three-input kernel directly for all
LayerNorm+ReLU sites without a residual skip. This removes the remaining
production-only fourth input/ref and kernel wrapper. The residual
LayerNorm+skip+ReLU path is deliberately unchanged; if no-skip stages become
exact, the first mismatch should move to the first skip stage and isolate its
additional operation.

