# Final history handoff: remaining integration contract

Read-only source audit, not CUDA execution:

- `D:/100XH100/src/history.cpp` SHA-256
  `8389834065024e7299f9999a6c072b258921519f87b48c2e2aee79a70e203d78`.
- `src/history.hpp` SHA-256
  `0221ba1a9ce1dd061671434cf3cbdbd118eec5ef612a23b5bd02fc869ad6f0753`.
- `cuda/final_materialize.cu` builds `FinalHistoryRecord` from the original
  candidate and target-local index, then scatters metadata at that index.
- `cuda/dispatcher.cu` checks received history count equals local target count
  and synchronizes the chunk pipeline before host history copy/event recording.

The TPU final request/response primitives do not yet implement this handoff.
Metadata must follow the destination frontier's indexing, not source-send
ordering or compactor ordering. State and history need identical target IDs.
Counts alone cannot prove uniqueness: duplicate targets plus a missing target
may preserve the count. Integration tests must compare every target identity
and retain parent high words and route bits.

The simple CPU `CpuHistoryStore` implementation retains full CandidateMeta
and reconstructs by indexing prior layers with parent_idx. It does not itself
consult source_rank while indexing. Do not assume that implementation alone
defines distributed rank-local history reconstruction: caller/rank handling
must be audited before adopting a compressed store for this port.

Acceptance still requires multidepth reconstruction and actual puzzle replay,
including balanced relocation, empty ranks, different parent/source/target
ranks, and K1/K2 solved paths that bypass normal final selection. A local
metadata copy test is insufficient. Host copy completion is also a lifetime
dependency before scratch reuse; no asynchronous-overlap claim without a trace.

## Distributed caller resolved

`D:/100XH100/tools/production_runner.cu` SHA-256
`45e2bde259be4763f867d39a363799b1e2aae6070cf52ff12fde057abc3461f7`
implements the missing rank interpretation in `reconstruct_solution_distributed`
(observed lines3420 onward). For each backward step it emits the cursor move,
then uses `unpack_source_rank(cursor.route_packed)` and cursor parent_idx to
query that rank's prior layer `out-1`. That rank returns parent_idx and route;
the new cursor determines the next rank. Queries and responses are coordinated
broadcasts. Source rank is therefore not the current history record's owner
and must not be rewritten during final balancing. Bounds are checked before
rank access. At the final root step there is no previous-layer query.

After reconstruction the caller appends solution suffixes separately. A TPU
host history service may change transport implementation, but must preserve
this rank/layer/local-index chain and suffix order. This resolves the semantic
gap in the simple CpuHistoryStore; it does not implement the TPU history path.
