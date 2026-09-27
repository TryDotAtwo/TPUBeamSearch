# Prepared eight-TPU final-depth execution gate

`benchmarks/beam_final_depth_execution.py` is a physical correctness runner.
It has not run on TPU yet. Run it only after the response epoch and state-width
bridge gates reach terminal status, preserving one Kaggle TPU session.

Each of eight ranks owns one tied, selected A0 candidate. Its parent lives on
the next rank, so all eight request/response paths cross TPU boundaries. The
runner composes frozen resident selection, global cap/balance, paired request
and history exchange, uniform epochs, Pallas materialization, response/history
coverage and the private frontier result. It saves JSON before lowering,
compilation and execution, plus lowered MLIR and compiled HLO when available.

Expected outputs are generated independently from the fixture: rank `r`
receives the state of rank `(r+1)%8`, has one target and a matching parent
history record with the original source route. The gate checks every frontier
byte including cleared response-index/padding, every history word, all eight
per-device errors and target counts, global keep/phase counts, dtype, shape and
SHA-256. It does not measure time or validate S1-S5 admission/stop.

The private Kaggle launcher is prepared under
`kaggle_beam_final_depth_execution/`, pinned to source
`b4ade889bdebb9b6020468709d490698c3768058`. It is not submitted yet.
The local pure fixture test establishes only input/oracle consistency;
eight-device correctness requires the physical run.

A fresh eight-device CPU JAXPR trace of the prepared `shard_map` wrapper
completed on 2026-09-27. Its flattened result shapes are frontier
`[8,128,160]`, tiled history `[8,1,5,128]`, five controls with the expected
leading eight-device axis, target counts `[8,8]`, keep/counts `[8,2,128]`,
and local coverage errors `[8,1,128]`. This checks the wrapper's static ABI
only; it does not compile or execute Mosaic on TPU.
