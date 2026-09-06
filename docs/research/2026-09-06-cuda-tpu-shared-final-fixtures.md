# Shared CUDA / physical TPU final-materialization gate

Prepared `benchmarks.beam_cuda_final_probe` consumes the six immutable binary
fixtures and actual CUDA outputs in `test_results/cuda_final_oracle_v2`.
It does not regenerate equivalent-looking random inputs. The loader checks
individual CUDA success, input/output SHA-256, binary header, geometry, counts,
exact byte lengths and duplicate case identities before allocating TPU inputs.

Each of eight physical TPU devices receives the same saved fixture. The unchanged
Pallas materializer must match every CUDA output byte, report zero invalid
requests, and zero every padded output row. Per-device valid-output SHA-256 must
equal the CUDA SHA. The second validation-summary row is the first-invalid-index
sentinel, not an error count.

This is a per-device materialization correctness gate, not distributed exchange,
multi-depth beam replay, or a performance comparison. No timing claim is made.
HLO, partial JSON, source SHA, runtime versions and device inventory are retained.

Local validation: seven fixture-loader tests pass, including corrupt hash,
trailing bytes, inconsistent count, duplicate case, corrupted output and failed
CUDA case rejection. The benchmark CLI imports successfully. Physical TPU
compilation/execution is **pending**. No production primitive or default changed.

Launch only after the current S5 V8 session reaches terminal state. Pair this
probe with the already prepared final-exchange probe in sequential isolated
processes, never a second concurrent TPU session.
