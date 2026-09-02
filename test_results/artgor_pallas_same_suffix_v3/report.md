# Same-suffix v3: input operators and attribution correction

Runtime: eight `TPU v5 lite` devices, JAX/jaxlib 0.10.2, libtpu 0.0.42.1.
Source: `09fd660f22d075213e93f062a7ab4f19630b3c4c`.
Checkpoint/model hashes match the recorded v3 context. Scope: 256 states/device,
legal seed 42 and categorical stress seed 43; no speed or 16K/32K promotion claim.

| Operator | Hidden mismatches legal/stress | Shared suffix Q mismatches |
|---|---:|---:|
| Embedding | 0 / 0 | 0 / 0 |
| Input Dense | 0 / 0 | 0 / 0 |
| Input LN vs prefix boundary | 21982 / 24861 | 9624 / 27957 |

Embedding and isolated input Dense match their references. But the LN row is
NOT yet causal attribution: its candidate consumes a separately compiled Dense
tensor while its reference `hidden[0]` comes from the whole JAX input prefix.
Likewise v2/v3 residual references are prefix outputs rather than an isolated
JAX block applied to the identical hidden input. A shared suffix does not remove
this upstream confound. The earlier claim that every isolated residual is
inexact is therefore withdrawn pending matched-input controls.

v4 applies both JAX and Pallas operators to the same runtime input tensor,
records isolated-reference versus prefix drift separately, and adds an explicit
zero-replacement suffix check. Arithmetic is unchanged until those results.
