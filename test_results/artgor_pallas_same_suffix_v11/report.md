# v11: three Pallas mean trees pass the two-corpus diagnostic

Source `71caa55191bad6733756572b29d2c66093dccedf`; eight TPU v5 lite,
256 states/device, JAX/jaxlib 0.10.2, libtpu 0.0.42.1. Checkpoint, model-source,
input hashes and native-prefix output hashes match v10.

All five candidates compile and execute, and all candidate outputs are finite.
There are no compile rejections. Mean mismatch counts are broadcast over 1024
columns, so 1024 means one differing row.

| Corpus | Order | Mean vs JAX | Prefix | Same-suffix Q |
|---|---|---:|---:|---:|
| legal42 | native | 0 | 0 | 0 |
| legal42 | lanes_serial | 0 | 0 | 0 |
| legal42 | lanes_tree | 0 | 0 | 0 |
| legal42 | tiles_serial | 0 | 0 | 0 |
| legal42 | tiles_tree | 0 | 0 | 0 |
| stress43 | native | 1024 | 15 | 17 |
| stress43 | lanes_serial | 1024 | 15 | 17 |
| stress43 | lanes_tree | 0 | 0 | 0 |
| stress43 | tiles_serial | 0 | 0 | 0 |
| stress43 | tiles_tree | 0 | 0 | 0 |

`lanes_tree` combines corresponding positions in 128-wide tiles through a
balanced addition tree, then reduces the resulting 128 values. The two `tiles`
variants reduce each tile first, then add tile sums serially or as a balanced
tree. Three candidates reproduce every BF16 mean and the complete prefix SHA
on both tested corpora; native and lane-serial retain the known single-row
failure. This establishes working Pallas reduction candidates, not a unique
reconstruction of the physical JAX reduction tree.

## Scope and next gate

These runs use materialized reference embedding, Pallas raw Dense, candidate
Pallas mean and the same Pallas remainder. The embedding Pallas primitive was
independently exact, but that does not prove one compiled all-Pallas prefix.
Next compose embedding + Dense + mean + remainder into the actual sharded
Pallas prefix and test all three surviving candidates against the unchanged
JAX prefix at 16K and 32K states/device on legal42/142/242 and stress43/143/243.
Keep original full Q/same-suffix controls separate: they are not identical.
Do not run the full old operator sweep at these sizes merely to validate the
prefix; prepare a dedicated bounded prefix gate with hashes/finite/counts/HLO.

There are no timing measurements here and no fastest-candidate selection.
The full model, six-corpus gate and large-batch gate remain unconfirmed.
No production default, BN, beam search or working hybrid was changed.
