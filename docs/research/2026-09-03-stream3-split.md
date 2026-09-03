# Bounded Stream3 split and V4 count-lowering correction

Accepted whole-port contract remains `TPU_ARCHITECTURE.md`. This is a bounded
diagnostic implementation, not the scalable HBM partition or a completed S3.

Inputs: unique candidate SoA uint32[8,N] in survivor order, owners uint32[1,N]
computed after dedup, valid_count uint32[1]. N is power-of-two 128..4096.
The caller guarantees count <= N, valid owner IDs and legal move IDs.
Static topology permits 1..256 owners and local_rank < world_size.

Outputs: separate local and remote SoA [8,N], local_count [1,C], send_count and
send_offset [1,C], C=round_up(world_size+1,128). Count entries 0..W-1 and offset
entries 0..W are meaningful; control padding is zero. The local peer has zero
send count. `send_offset[W]` is the remote valid count. Data tails have zero
metadata except score UINT32_MAX and are invalid regardless of threshold.

Inside Pallas, pack source/owner/move route; stable bitonic partition for local
using validity/original position, remote using validity/owner/original position.
There is no host data-dependent selection, atomic append, scatter or owner cap.
Two independent full-capacity outputs are deliberately retained for a transparent
baseline; this is not yet a scratch aliasing or memory-efficiency claim.

Counts use signed int32 sums then uint32 output. Logical scalar counts occupy
element [0,0] of an aligned control plane and all remaining lanes are zero; a
Pallas scalar VMEM store is not emitted. The bound N<=4096 proves exact
conversion, avoiding V4's explicit Mosaic rejection of unsigned reductions.
Per-peer offsets are a static sequence of bounded additions. No inter-device
operation occurs in this primitive; local_rank is static for this diagnostic.
Wiring rank-varying sharded callers is a separate integration step.

Verification includes empty/nonempty, world=1/8, stable peer order, parent high
words, tail neutralization, zero-count peers, and original C++ Stream3 output.
Two standalone eight-device cases join the next isolated Kaggle bundle. They
use replicated fixtures (static rank 3) to prove compilation/values, not actual
rank-to-rank delivery. Physical compilation is outstanding.

Unresolved S2 flatten/layout errors remain unchanged in the same bundle.
Expert requests returned network error/timeout and supplied no usable advice;
no design or correctness claim relies on expert endorsement.
