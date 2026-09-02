# Exact-probe all-Pallas diagnostic v8

Private Kaggle kernel `trydotatwo/tpu-artgor-all-pallas-exact-diagnostic` v8
completed from source `dc7f91caaf69e30ec423a7c6927e2f9900cfd6aa`
on eight TPUv5lite devices.

Directly reusing the standalone TPU-proven exact no-skip LayerNorm probe still
produces the same 21,165--25,312 reported mismatches at
`input.layernorm_relu`. Since this is now literally the same Pallas call that
was hash exact in the isolated monolithic-match experiment, the discrepancy is
in the reference trace, not the candidate kernel.

The stage oracle returns all 44 JAX intermediates from one compiled multi-output
function. Materializing those extra outputs changes the JAX LayerNorm lowering;
it is not the unchanged `jax_model.apply` arithmetic contract. Therefore this
instrumented trace remains useful for localization but is not a valid gate for
the final model.

V9 always evaluates the candidate against the unchanged full
`jax_model.apply` output at B16K/device and, when exact, B32K/device. It retains
the 44-stage trace as non-gating diagnostics. Promotion still requires exact
full BF16 Q, clean all-Pallas HLO and measured speed over `exact_split`; default
selection remains blocked until a separately valid operator-boundary audit is
available.

