# K1/K2 and bounded solved publication: source audit

Source remains read-only. `cuda/stream2.cu` SHA256:
`d52252daba39fc913a31c7ded25f08721b8b58aa8c5ca0e09872bf7a6e61f126`.
Architecture requirements: `ARCHITECTURE_NEED.md:425..481`.
This note is not an implemented K1/K2 claim.

## Two different hashes must survive

Stream2 always writes the immediate parent+move Hash128 into hash_ring before
checking K2. A K2 hit changes solved_meta.hash to the suffix-projected hash,
but must not replace hash_ring with that hash. S3 still routes/deduplicates the
immediate child. The solved record retains the immediate move/source and
parent identity plus a separate suffix_id. History reconstruction appends K2
suffix moves first, then the CPU K1 suffix found by the solved hash.

## Lookup and suffix order

K1=0 uses full central-state comparison. K1>0 uses two source hash buckets,
four slots each: fingerprint is only a prefilter, followed by full Hash128
equality. GPU table contains no states or suffix chains. CPU inverse-move
neighborhood generation and table packing need separate source parity tests.

Immediate/direct-K1 hit wins with suffix_id=0. Only if it fails does the kernel
try suffix IDs1..count-1 in order, returning the first hit. The base-generator
backend walks a packed suffix in reverse while composing source indices; the
composed-permutation backend supplies that projection directly. Both then
compose the immediate move before reading parent values. Reversing the wrong
part changes the state, so compare the two backends on noncommuting moves.

The CUDA packed suffix decoder uses5bits per move in uint64. The generic
MOVE_COUNT<=255 metadata constraint does not prove a suffix fits that format.
Before porting the builder, locate and preserve its move/radius guards; do not
silently mask arbitrary move IDs into5bits or shift beyond packed capacity.

## Solved collection and stop

The source increments solved_count for every attempted hit, including attempts
beyond capacity; solved_overflow flags lost storage. The count is not just the
number of stored records. It stores meta/depth/suffix before fence/solved_flag
publication and sets stop only when stop_on_found is enabled. Active kernels
may still report additional hits. Solved data and stop live outside scratch.

TPU may use prefix compaction and bounded reservations instead of this CUDA
atomic path, but must preserve attempted/stored counts and overflow, retain
all accepted records and coordinate stop across ranks without unilateral exits
from active exchanges. Goal hits bypass beam threshold/final selection.

Remaining acceptance: original CPU builder/lookup/suffix oracle; physical
TPU lookup including fingerprint collisions; both suffix backends; repeated
and overflow hits; multi-rank stop/drain; replay reconstructed full solutions.

## Located host builder and concrete guards

The builder lives in `tools/production_runner.cu`, inspected SHA256
`45e2bde259be4763f867d39a363799b1e2aae6070cf52ff12fde057abc3461f7`.
K1 host radius is limited to12, with a startup max-entries guard. K2 direct-scan
runtime radius is limited to3; the generic packed-suffix helper allows12.
Packed capacity is not permission to run a direct scan of radius12.

K2 enumeration is breadth-first: empty suffix, then each prior suffix in order
and move0..MOVE_COUNT-1. It retains all chains, not unique resulting states.
Radius3 gives14425 suffixes for24moves and27931 for30moves. The optional count
guard raises before an append exceeding the limit; the final count must fit
uint32. Preserve suffix IDs and first-hit ordering in the TPU builder.

The inspected helpers check move<MOVE_COUNT and length<12, but do not themselves
check move<32; decoding masks5bits. Reject MOVE_COUNT>32 for the TPU packed
suffix ABI instead of truncating, unless a wider representation is explicitly
designed and tested. This startup guard does not modify read-only CUDA source.
Current24/30-move workloads fit.

## Immutable TPU table preparation

`prepare_k2_suffix_table` now prepares three128-aligned uint32 planes: packed
moves low/high and length, with an explicit valid count. Padding is never an
additional empty suffix. Entries preserve source BFS chain order (no state
dedup), and the first move occupies the lowest5bits. Geometry checks reject
MOVE_COUNT>32 and direct radius>3 before allocation. The max-count guard checks
the exact geometric count before allocating. Radius0 returns one canonical
empty entry; runtime disabling remains a separate caller decision.

The missing-module test was red; eight host tests pass in2.97 s, including
manual two-move order,24/30-move counts14425/27931, padding and bounds. Artifact:
`test_results/local_suffix_table_regression.xml`. This is source-inspected host
preparation, not execution of the original C++ builder, not a Pallas suffix
scan, and not K1/K2 CUDA/TPU acceptance. Device lookup, first-hit selection and
full reconstructed-solution replay remain outstanding.

`pallas_suffix_projection` now composes source-index permutations from those
tables, using the existing Stream2 gather helper. It reverses suffix step
indices, yielding the same forward state transformation; the immediate move
must still be composed afterwards by Stream2. Output is transposed int32
`[state_width,suffix_capacity]`, with zero invalid columns. Validated generator
indices and length<=3 are caller contracts. This prepares the optional composed
backend; it does not search for hits or change immediate-child hashes.

After a missing-module red test, noncommuting two-move and radius3/156-suffix
cross-tile interpreter cases pass against explicit forward NumPy application.
The combined table/projection suite passes10 tests in3.53 s:
`test_results/local_suffix_projection_regression.xml`. Physical Mosaic
compilation, real24/30-move memory/timing and K1/K2 integration remain open.

## K1 fingerprint and bucket keys

`pallas_k1_keys` reuses the uint32-pair distribution arithmetic with the source
fingerprint/bucket salts. Fingerprint is low XOR high, mapping0 to1; both bucket
keys take low32 and mask by bucket_count-1. Original `src/hash.hpp` SHA256:
`361756fe2de60ae9393f0e60f6be80c697e9b84c58fbaedbc75d1c5d8162016c`.

The local C++ oracle adapter now exposes `k1keys` calling those original inline
functions, without modifying D:/100XH100. After a missing-module red test,
256 edge/random hashes match for1/32/1024 buckets, including a constructed zero
fingerprint preimage. The combined new/source parity tests pass9 in32.23 s:
`test_results/local_k1_keys_regression.xml`. This is original C++ on CPU versus
Pallas interpreter, not CUDA or physical TPU execution. Actual four-slot,
two-bucket lookup and full Hash128 collision checking still need implementation.

## Bounded lookup and fixed table preparation

`pallas_k1_contains` now reads aligned128-slot windows from HBM into a5x128
uint32 scratch window and checks exactly four slots of each source bucket.
Fingerprint equality must be followed by all four Hash128 words matching.
Query count masks padding; it uses two serialized table reads per valid query,
not an optimized coalesced lookup. Seven interpreter cases pass in12.35 s,
covering both bucket paths, fingerprint collision, all four slots and exclusion
of DMA-window slots outside the bucket. Tests require TPU InterpretParams;
generic interpret=True failed on CPU program_id lowering. No physical TPU
lookup acceptance is claimed. The module is included in regression82378.

`prepare_k1_table` implements source fixed-arena placement: first empty slot of
bucket0, then bucket1, preserving entry order. It raises on failure, never grows
or drops entries. Three host tests pass in4.99 s, including a bucket collision
with unused global slots; artifact `test_results/local_k1_table_regression.xml`.
Placement is checked with Pallas keys already compared to original C++ helpers;
the original host packing function itself was inspected, not executed. This
new table module was added after regression82378 collection. Inverse-neighborhood
BFS, suffix-by-hash storage and reconstructed-solution replay remain pending.

The fixed table and HBM lookup are now tested together, including a legitimate
zero Hash128 and an empty table. All11 table/lookup tests pass; the recorded
artifact is `test_results/local_k1_integration_regression.xml`. This does not
replace the missing inverse-neighborhood builder or physical lookup gate.

`prepare_k1_neighborhood` now builds the inverse BFS, clears state padding,
retains first visits by Hash128 and prepends the move to the suffix chain.
It returns an immutable suffix mapping and fixed K1 table. Two initial host
tests pass in2.86 s after a missing-module red test: six manually enumerated
states/first suffixes, replay to central, max-entry failure, disabled radius0,
and forced Hash128 collision deduplication. Artifact:
`test_results/local_k1_neighborhood_regression.xml`. This is not original C++
builder execution or physical K1/K2 acceptance. Radius<=12 and five-bit move
IDs are validated; device hit selection/solved collection still need wiring.
This module and table preparation were added after regression82378 collection.

The combined K1 keys/lookup/table/neighborhood and K2 table/projection suite
passes26 tests in22.04 s with the original-source CPU oracle enabled for keys:
`test_results/local_k1_k2_preparation_regression.xml`. The additional generated
neighborhood-to-Pallas-lookup test then passed in the three-test neighborhood
suite (5.18 s), including rejection of an outsider. Full collector regression
82378 completed764 tests in1066.95 s, no failures/errors/skips; later table/BFS
additions have the separate evidence above. Physical acceptance remains pending.
