"""Compact one complete remote exchange into the CUDA receive batch order.

Snapshot/external-sort baseline, not bounded two-slot streaming or DMA overlap.
Inputs contain peer-offset epochs, excluding the local-owner input, which is
collected separately. Caller must preserve snapshot ownership through this call.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_external_sort import pallas_external_bitonic_sort


def pallas_compact_received(snapshots,counts,receiver_rank,*,interpret=False):
    """Epoch e came from (receiver_rank-e-1) % world_size; counts mask tails.

    Valid count <= snapshot capacity and rank < world_size are caller contracts.
    Output uses power-of-two capacity, preserving all eight metadata planes.
    """
    if (snapshots.ndim != 3 or snapshots.shape[1] != 8
            or not 1 <= snapshots.shape[0] <= 255
            or snapshots.shape[2] < 128 or snapshots.shape[2] % 128
            or counts.shape != (snapshots.shape[0],128)
            or receiver_rank.shape != (1,128)):
        raise ValueError('invalid receive batch geometry')
    if any(x.dtype != jnp.uint32 for x in (snapshots,counts,receiver_rank)):
        raise ValueError('receive batch requires uint32')
    epochs,_,capacity = snapshots.shape
    n = 1 << (epochs*capacity-1).bit_length()
    if n > 16384:
        raise ValueError('diagnostic receive batch exceeds external-sort bound')
    tiles = capacity//128
    def prepare(x,c,r,out):
        epoch = pl.program_id(0)//tiles
        safe_epoch = jnp.minimum(epoch,epochs-1)
        idx = (pl.program_id(0)%tiles*128+jnp.arange(128,dtype=jnp.int32)).astype(jnp.uint32)
        valid = (epoch < epochs)&(idx < c[safe_epoch,0])
        sender = (r[0,0]+jnp.uint32(epochs+1)-epoch.astype(jnp.uint32)-jnp.uint32(1))%jnp.uint32(epochs+1)
        out[...] = jnp.concatenate((x[0],valid[None].astype(jnp.uint32),
            jnp.broadcast_to(sender,(1,128)),idx[None]))
    spec = pl.BlockSpec((11,128),lambda t:(0,t))
    data = pl.pallas_call(prepare,out_shape=jax.ShapeDtypeStruct((11,n),jnp.uint32),
        in_specs=(pl.BlockSpec((1,8,128),lambda t:(jnp.minimum(t//tiles,epochs-1),0,t%tiles)),
                  pl.BlockSpec(counts.shape),pl.BlockSpec(receiver_rank.shape)),
        out_specs=spec,grid=(n//128,),interpret=interpret,
        name='beam_receive_epoch_prepare')(snapshots,counts,receiver_rank)
    data = pallas_external_bitonic_sort(data,key_planes=(8,9,10),validity_plane=8,interpret=interpret)
    def finish(d,out):
        neutral = jnp.where(jnp.arange(8,dtype=jnp.int32)[:,None] == 6,
                            jnp.uint32(0xffffffff),jnp.uint32(0))
        out[...] = jnp.where((d[8] != 0)[None],d[:8],neutral)
    packed = pl.pallas_call(finish,out_shape=jax.ShapeDtypeStruct((8,n),jnp.uint32),
        in_specs=(spec,),out_specs=pl.BlockSpec((8,128),lambda t:(0,t)),
        grid=(n//128,),interpret=interpret,name='beam_receive_batch_finish')(data)
    def total_kernel(c,out):
        total = jnp.uint32(0)
        for e in range(epochs):
            total += c[e,0]
        out[...] = (jnp.arange(128,dtype=jnp.int32)[None] == 0).astype(jnp.uint32)*total
    total = pl.pallas_call(total_kernel,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        in_specs=(pl.BlockSpec(counts.shape),),out_specs=pl.BlockSpec((1,128)),
        grid=(),interpret=interpret,name='beam_receive_batch_count')(counts)
    return packed,total


def pallas_collect_received(a,b,wire_output,controls,counts,receiver_rank,*,interpret=False):
    """Consume the exchange ABI: two reusable slots followed by epoch snapshots.

    ACK already follows snapshot completion in this transport. This collector
    reads snapshots, never the reusable slots. One complete exchange is one
    admission; full-result synchronization is still required before S4 use.
    """
    from .beam_collector import pallas_collect
    if wire_output.ndim != 3 or wire_output.shape[0] != counts.shape[0]+2:
        raise ValueError('wire output must contain two slots plus all snapshots')
    packed,total = pallas_compact_received(wire_output[2:],counts,receiver_rank,interpret=interpret)
    return pallas_collect(a,b,packed,controls,total,interpret=interpret)
