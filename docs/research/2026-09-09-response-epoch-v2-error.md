# Response epoch V2: zero-length scan slice rejected

Private Kaggle version2, source `7c62221c19343dc385539755476cfb0484a43f43`,
launcher `677a69a`. Output retrieved after network recovery into
`test_results/beam_response_epoch_v2_retry/`; the first downloader remains
pending in a separate directory and is not evidence of another TPU run.

Process returncode1. JAX/jaxlib0.10.2, libtpu0.0.42.1; eight distinct device
IDs0..7, all TPU v5 lite. Partial JSON contains only pending empty case,
zero epochs, exact=false. There are no timing or successful execution claims.

`prep.lower(*args)` fails at `beam_final_intervals.py:40`:
`lax.associative_scan(jnp.add,counts)-counts`. Mosaic reports a slice whose
vector type has size0: "vector types must have positive constant sizes but
got 0". This is a Python/MLIR lowering exception, not a native abort.
V1's unsigned reduce_sum rejection is no longer the reported failure; that
does not establish numerical or execution acceptance of the repaired path.

Next: reproduce the zero-sized intermediate in a structural regression,
replace the fixed128-lane exclusive prefix calculation with a lowering-safe
equivalent preserving uint32 ABI and count bounds, and test literal interval
expectations including empty ranks, high rank, malformed rank and multiple
tiles. CPU/JAXPR checks cannot clear the TPU gate. Full regression, public
source pin and a single terminal-state successor run remain mandatory.
