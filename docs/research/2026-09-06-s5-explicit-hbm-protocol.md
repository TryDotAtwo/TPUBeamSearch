# Explicit custom-call HBM output diagnostic

V3 and V5 communicating outputs carry S(1) plus a root copy; V4 local-only
output does not. Initialization did not fix V5. This motivates a placement
control, not a production fix declaration.

JAX 0.10.2 source independently confirms that `_resolve_memory_spaces` reads
the pallas call's `out_avals` and maps a TPU HBM tag to the custom-call memory
constraint. This is distinct from the kernel's BlockSpec placement:
https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/pallas/mosaic/pallas_call_registration.py
(functions `_get_memory_space_from_aval`, `_resolve_memory_spaces`, and
`_lower_to_custom_call`). Local tracing confirms the tag is retained in call
parameters, while the outward JAX aval is the generic device type.

Run two sequential isolated subprocess groups in one private TPU session:

- hbm: unchanged wire body, explicit `pltpu.HBM` output type.
- hbm_initialized: same explicit type, plus V5's full local initialization.

Both use identical public seed603 wire fixtures: eight nonzero/zero/nonzero/
singleton cases, two repetitions, same peer-offset expected identities.
No change to payload, ranks, dimensions, semaphore sequence, or default path.
Coordinator must run second group even if the first fails. Save each HLO,
runtime and source identity, complete logs and failure NPZ.

Acceptance requires eight physical devices and all eight cases exact with
matching hashes in each accepted group. A changed HLO layout without correct
data does not pass. If only initialized passes, the placement constraint alone
is insufficient. If neither passes, preserve the result and investigate
another hypothesis. No timing or integrated S5 claim from this diagnostic.

Local targeted suite:31passed3.61s. Full suite session25104 started with both
source-oracle environment variables set; wait for terminal result before push
and pinned launcher submission. TPU V6 has not been submitted at this point.
