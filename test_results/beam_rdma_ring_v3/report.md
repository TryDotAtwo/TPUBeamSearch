# Stream3 variable-count RDMA gate V3

Private Kaggle kernel `trydotatwo/tpu-beam-rdma-ring-probe` ran source
`ba569bb05a40ad434572a1d6c4fecb3280571b9c` on TPU and failed during Pallas
lowering before execution.

The exact failure is a direct scalar load from `count_ref[epoch, 0]` whose
BlockSpec uses `pl.ANY` (HBM). Mosaic permits direct loads only from VMEM/SMEM;
HBM/ANY references must be accessed through asynchronous copies. The received
count had the same latent issue at `count_out[epoch, 0]`.

The minimal correction keeps payload and wire-count planes in HBM, passes a
duplicate count tensor through `PrefetchScalarGridSpec` scalar prefetch for the
local send predicate, and DMA-stages each received count plane into a two-slot
VMEM buffer before evaluating the receive predicate. Payload protocol and
fixed-capacity wire ABI are unchanged. V4 must physically confirm this lowering
and then check all data/count hashes.

