# Final streaming source contract

Read-only CUDA audit at `b5fcf6b0ee3247a1f189e6da79a1641d2304bd1c`.
This is source evidence, not executed CUDA/TPU parity or a performance result.

## Observed implementation

- `cuda/dispatcher.cu:4315`: all ranks use rounds derived from the maximum
  selected count and the configured chunk capacity. Each round slices a
  contiguous local selected prefix, not an independent chunk per destination.
- `dispatcher.cu:4390-4419`: three rotating slots retain pending responses;
  reuse drains the response event and scatters received states first.
- `dispatcher.cu:4452-4470`: history exchange begins before request exchange.
  `4525-4534` waits for history and scatters records directly into final
  candidate storage. No whole-response-wire log is accumulated.
- `dispatcher.cu:4504-4560`: local-source requests bypass network exchange;
  local plus remote work is explicitly bounded by exchange capacity.
- `dispatcher.cu:4586-4633`: return-rank grouping, direct local materialization,
  and one variable-size remote response exchange per selected chunk round.
- `dispatcher.cu:4639-4645`: aggregate history and response receive counts must
  equal the local target count, then stream3 is synchronized before host copy.

## Uniqueness is not a scatter-time duplicate check

`cuda/final_materialize.cu:116-157` derives global indices from disjoint
less/equal rank-prefix intervals and maps these to balanced rank/local target
indices. This construction, exact selection and lossless transport establish
the intended one-to-one mapping. `final_scatter_history_records_kernel`
(line210) and `final_scatter_responses_kernel` (line271) directly index their
outputs; neither implements a duplicate bitmap or a missing-target scan.
Equal total counts alone would not detect one duplicated and one missing
target. Do not describe the source count check as an exact coverage check.

## Consequences for the in-progress TPU caller

The existing TPU per-peer128 request epochs differ from CUDA's total selected
chunk rounds. One such epoch can produce world*128 responses at a source,
possibly all for one return rank. Uniform world response subrounds are a safe
static upper bound, not CUDA's measured or optimal schedule. Empty/error ranks
must participate. The currently serialized request-then-history order is also
an explicit scheduling difference; no overlap has been demonstrated.

Retain private target-indexed frontier/history storage and bounded transport
scratch. Do not accumulate all wire bytes merely to validate coverage. A
scalable exact target-coverage structure, or a separately justified invariant
plus diagnostic coverage mode, still needs implementation and physical tests.
The existing whole-target HBM sort is a diagnostic gate, not evidence that
production streaming memory/lifetimes are complete. Invalid partial results
must never replace the old published frontier/history handle.
