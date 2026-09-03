# Composed prefix gate v1: all candidates rejected at large batch

Source `1e63659243ff66e3582ebeb522d509ee4ddad43e`; eight TPU v5 lite,
JAX/jaxlib 0.10.2, libtpu 0.0.42.1. Checkpoint and model-source hashes match
the previous diagnostic. All 36 cases completed with finite outputs; no
compile/execution errors. All six shape/order HLO audits show exactly four
Pallas custom calls and no forbidden Dense/gather/reduce/maximum operations.

Nevertheless, every case fails bitwise equality. Counts are BF16 elements,
not rows. Denominator is 134217728 elements at 16K/device and 268435456 at 32K.

| Batch/device | Corpus | lanes_tree | tiles_serial | tiles_tree |
|---:|---|---:|---:|---:|
| 16384 | legal42 | 1329 | 1321 | 1329 |
| 16384 | legal142 | 791 | 754 | 754 |
| 16384 | legal242 | 1913 | 2010 | 1963 |
| 16384 | stress43 | 1217 | 1251 | 1252 |
| 16384 | stress143 | 2160 | 2162 | 2189 |
| 16384 | stress243 | 793 | 758 | 749 |
| 32768 | legal42 | 2892 | 2930 | 2938 |
| 32768 | legal142 | 3466 | 3502 | 3503 |
| 32768 | legal242 | 2751 | 2931 | 2884 |
| 32768 | stress43 | 1547 | 1517 | 1532 |
| 32768 | stress143 | 4270 | 4260 | 4315 |
| 32768 | stress243 | 1589 | 1572 | 1561 |

The first failing case is 16K legal42; all three orders first differ at
[760,0]. No candidate is eligible. No tolerance or default is changed, and
there is no timing/speed result.

## Attribution boundary

This test changes both workload size and composition relative to v11. Larger
corpora may expose rare errors, and compiled layout may change. The JAX
reference HLO at 16K uses Dense/bias matrix layout `{0,1:T(8,128)}`, whereas
the 256-state diagnostic used `{1,0:T(8,128)}`. This is a concrete compiler
layout observation, not proof that it caused these discrepancies. Identical
seed labels at different batch sizes do not alone prove identical row streams.

Next use the SAME saved/generated 16K legal42 states with full-batch and
256-per-device chunked executions for BOTH JAX and Pallas. Compare JAX large
against JAX chunked, Pallas large against Pallas chunked, and matched-size
JAX/Pallas. Keep per-device row ordering and hashes explicit. Then perform
same-input embedding/Dense/mean substitutions at the first divergent boundary,
capturing affected rows. Do not expand residuals until input-prefix equality
holds under the required workload. Original full-model Q remains a later,
separate gate; the prefix reference is not a replacement for that oracle.
