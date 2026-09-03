# TPU beam architecture

Status: accepted direction, implementation in progress. Source contract:
`D:\100XH100\ARCHITECTURE_NEED.md` (1885 lines; file history commit
`894095f2`). The source remains read-only. This is the TPU implementation
contract, not a claim that the full search is implemented.

Whole-port progress and outstanding gates: [TPU_PORT_LEDGER.md](TPU_PORT_LEDGER.md).

## Logical types versus storage

- State values remain uint8. Separate logical state width, persistent storage
  width (16-byte alignment, space for a uint32 response index), and kernel
  tile width. Persistent padding is zero; response index occupies the first
  four bytes after STATE_LEN and is cleared before frontier insertion.
- Candidate metadata is a flat uint32 SoA with eight planes, each with aligned
  candidate capacity. Planes 0..3 are Hash128 least-significant word first;
  4..5 are parent_idx low/high; 6 is score_key; 7 is route_packed. This retains
  all 256 bits of the CUDA record, not its AoS layout. Sort Hash128 comparing
  planes 3,2,1,0 (unsigned). Parent tie-break compares 5,4. Never cast these
  identities to float or truncate the parent's high word.
- route_packed retains source_rank:16 / owner:8 / move:8. Reject unrepresentable
  ranks and MOVE_COUNT rather than silently masking. Counts and offsets are
  uint32 only where allocation bounds prove safety; accumulated histogram
  counts require carry-aware uint32 pairs or verified uint64 support.
- Validity is explicit (count/mask). UINT32_MAX is the invalid score sentinel,
  but a threshold of UINT32_MAX does not make invalid slots valid. Hash zero
  and parent zero are legal. Padding actions never become candidates.
- Score semantics remain FP32 clamp/scale/round-to-nearest-even. NaN policy
  must be explicit at inference integration; finite is a required model gate.
- Physical SoA padding is not additional records and is included in HBM/DMA
  accounting. Tile legality and alignment depend on dtype and actual runtime;
  an aligned candidate plane does not establish legality of every state kernel.

## Stages and ownership

S1 writes one MOVE_COUNT score vector per parent, without an HBM q_float array.
S2 writes child Hash128 and goal records, not a full child frontier. S3 performs
threshold, compaction, Hash128 sort/dedup, owner routing; only its collector
writes survivor buffers. S4 performs threshold/compact/sort/dedup and committed
histograms; never per-shard top-k. Each logical shard has two resident physical
buffers; capacity is independent of dirty-work launch batch size. No spill path.
When neither writable sibling fits, record fatal overflow, never discard data.

S5 exchanges grouped metadata, not parents; it owns coordinated periodic global
histogram refresh. All ranks execute identical collective epochs even with no
local work. The source's 2026-05-26 Stream5 update supersedes older local-only
multi-rank threshold text. Threshold never relaxes after initialization.
Dedup tie-break is score/payload in S3, score/parent/route in S4.

Final drains pending work, deduplicates across A/B siblings, computes global
threshold and exact beam cap including ties, balances, exchanges parent
requests and child responses in chunks, clears response padding and emits CPU
history. Goal records bypass threshold/dedup/final selection. K1/K2 lookup and
suffix reconstruction remain explicit port requirements, not disabled defaults.

## Overlap from the first implementation

Use Pallas HBM-to-VMEM pipelines with two buffers initially; compare three input
buffers when the complete VMEM live set fits. JAX 0.10.1 rejects more than two
output buffers in emit_pipeline; the first packer keeps its output double-buffered.
Start the next DMA before computing the current
tile. Metadata SoA transport/packing is the first pipeline correctness fixture.
It is not evidence of overlap between independent S1/S2 compute kernels.

For inter-device overlap, use a bounded SPMD exchange epoch: publish destination
readiness, start remote copy to a free destination slot, compute independent
local work, wait for receive before consuming, wait for send before reusing
source, acknowledge consumption before reusing destination. Allocate separate
semaphores per live transfer slot; handle zero-count peers and stop collectively.
No unilateral early return around a collective. Ordinary asynchronous Python
dispatch is not sufficient to implement this protocol.

JAX is the compile/sharding shell; search data-plane kernels are Pallas.
First test the local pipeline and remote ring probe independently, then attach
real stages. Profile compute-only, transfer-only, serialized and overlapped
variants with identical work and outputs. Report startup/drain separately.

## Memory phases and gates

Persistent current frontier, immutable tables/weights and solved records live
outside the scratch arena. Streams, final selection and final materialization
are three mutually exclusive scratch layouts with explicit lifetimes. A phase
change requires all DMA and consumers drained. Buffer donation/aliasing must be
verified in compiled memory reports, not inferred from Python variable reuse.

Gate order: type/bit tests; Pallas interpreter; DMA/race simulation; real TPU
compile and differential per-stage checks; complete one-depth replay; 8-device
multi-depth replay with ties, overflow, uneven and empty ranks, K1/K2 and stop.
No speed claim from interpretation. Preserve BN and exact_split defaults.

References checked 2026-09-03:
- https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html
- https://docs.jax.dev/en/latest/pallas/tpu/distributed.html
- https://docs.jax.dev/en/latest/pallas/tpu/details.html

## Implemented foundation (2026-09-03)

`beam_types.py` implements validated host packing/unpacking and storage geometry.
`beam_transport.py` implements an HBM outer call and buffered VMEM packing
pipeline; metadata remains uint32 throughout. It is an isolated transport
primitive, not integrated into the production inference or beam.

18 focused tests pass on local JAX 0.10.1, including TPU InterpretParams with
race detection and an explicitly simulated TPU v5 lite abstract geometry.
Five distinct tiles exercise buffer wraparound for 2/3 input buffers, both
serialized (`no_pipelining=True`) and pipelined execution. This is
not physical TPU compilation, an overlap trace, or a throughput measurement.
