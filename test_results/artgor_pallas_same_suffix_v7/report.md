# Composition v7: remaining prefix-scope drift

Source `cbb5ef18f0be8ea1b4ddb02f291927bcd493e8d1`, eight TPU v5 lite,
256 states/device, legal42/stress43. Full all-Pallas still fails: 45926/54013
mismatched BF16 Q values out of 61440. No promotion or speed claim.

With late-skip arithmetic, residual0 hidden mismatches fall to 18377/17568;
same-suffix Q mismatches to 567/816. All ten composed blocks remain inexact.
These are not failures of the individually exact operator tests: compilation
of the JAX composite changes arithmetic at boundaries.

The JAX input-prefix compiled HLO has a multi-output Dense/bias/mean fusion.
It reduces the FP32 bias-add result and separately converts that result to BF16
for subsequent centering. Current Pallas materializes BF16 before computing
the mean. This is a concrete composition hypothesis, not a proved physical
rounding interpretation (v6 already showed why HLO text alone is insufficient).

v8 preserves raw FP32 preactivation in a diagnostic Pallas Dense and tests
dot-round-before-bias versus late dot rounding, crossed with FP32 versus BF16
mean input. Centering still uses BF16 values. It records rounded Dense equality,
standalone LN control, prefix boundary equality, shared-suffix and original-Q
comparisons. Hand-derived CPU/interpreter witnesses cover sub-BF16 bias and
distinct mean/centering sources. Production defaults stay unchanged.
