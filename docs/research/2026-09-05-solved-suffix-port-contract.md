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
