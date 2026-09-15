# Response layout repair: evidence and next probes

Status: investigation, not an accepted implementation or TPU speed result.
Receive full regression is running separately; Python sources are frozen.

## Byte conversion

The physical stage-isolation V1 log identifies the column store in
`pallas_planes_to_wire`, not the shift or mask, as the rejected operation:
`tpu.reshape vector<128xi8> -> vector<128x1xi8>`. Input is uint32 SoA
`[words, N]`; output is little-endian uint8 `[N, words*4]`. All padding bytes
must be preserved. Existing tests cover random bytes at widths128/256 and
high-bit target identifiers; these are interpreter checks only.

The [JAX TPU guide](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
documents restricted narrow-type slices and trailing-dimension reshapes.
This supports testing full rectangular stores instead of byte-column stores;
it does not establish that any proposed transpose or bitcast compiles on the
pinned0.10.2/.42.1 runtime. The guide's broad integer-reduction statement is
not a substitute for this runtime's observed signed/unsigned behavior.

Next candidate: construct a complete uint32 byte-value tile with explicit
little-endian shifts, cast only the complete rectangle to uint8, and store
the complete output window. Separately isolate any gather/transpose used to
map source planes to destination columns. Avoid replacing a known failing
store with an untested bitcast/reshape chain and calling it a repair.

Acceptance must include zero/UINT_MAX/0x01020304/asymmetric lane patterns,
both widths, multiple128-row tiles, every output byte including padding,
plus the unchanged response target consumer. Local equality is necessary;
physical compile and execution remain separate gates.

## Packing native abort

Packing alone aborts in VectorLayout::join (rank1 versus2); the exact source
expression is still unknown. Preserve original input shapes and runtime
chunk index when isolating these cumulative groups:

1. Zero outputs plus interval/prior-error predicate and control publication.
2. Add bounded runtime chunk offset and one-contributor peer selection.
3. Add guarded first DMA with start/wait and fixed-window output copy.
4. Add guarded second DMA with start/wait.
5. Add dynamic shift/gather and masked output/control writes.

Each group needs its own subprocess, pending report before compile, MLIR,
return code, log, and compiled HLO when available. Keep results live through
outputs so dead-code elimination cannot erase the expression under test.
If a cumulative boundary first fails, split that boundary further before
attributing the source cause. Changed layout can move the failure, so retain
the original packing stage and full composition as controls.

No new remote job has been submitted for these proposals. Expert dispatch
was rejected by the tool's disclosure review; no advice was received.

## Integration obligations after compiler repair

Read-only audit against `docs/TPU_ARCHITECTURE.md` and source architecture
lines1671 onward confirms that compiling response chunks is not completion:

| Boundary | Current evidence | Still required |
|---|---|---|
| Response transport | `make_final_response_chunk_call` composes packing, exchange, receive and byte conversion; returns private bytes | Physical execution of all response epochs before publication |
| Whole response coverage | `test_beam_final_agreement.py` covers129 valid rows, cross-chunk duplicate and late error, then scatter | Feed actual multi-rank epoch outputs, not preassembled bytes; preserve accumulated error |
| History | `test_beam_history_decode.py` covers parent64, original source and host all-rank atomic validation | Route actual history by balanced destination, finish transfers, couple commit decision to frontier |
| Scratch | `beam_scratch.py` plans three overlays and provides aliased writes | Actual stages must consume arena refs; read helper copies and cannot prove one-pool residency |
| Publication | Coverage agreement explicitly is not a complete barrier | Establish send/receive/consumer/history completion before publishing and before changing overlay |

The source places `next_frontier_states_tmp` inside final scratch and clears
response padding before insertion. Do not add another persistent full frontier
as a convenience while wiring the caller. Common error zero is a validation
result, not a DMA-drained token. Existing local fixtures do not establish
multi-device atomic publication or physical buffer aliasing.
