"""Serialized histogram all-gather plus exact pair sum; no overlap claim."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_histogram_pair_sum import pallas_sum_histogram_pairs


def make_s5_histogram_call(mesh,*,width,interpret=False,return_wire=False,own_only=False,local_replicate=False,initialize_wire=False):
    """All ranks enter after the common request decision with frozen local sums.

    Uses 2*ranks*width uint32 HBM scratch, not a ring reduce-scatter. The local
    input stays immutable until all sends complete. Each offset owns a distinct
    destination; no receive-slot reuse. Physical multi-rank validation pending.
    own_only isolates the unchanged local tile loop and returns [2,width],
    without remote transfers or reduction; diagnostic only.
    local_replicate retains full output shape and writes the local source to
    every pair slot, without remote DMA; all output regions are initialized.
    """
    ranks = mesh.size
    if own_only and local_replicate:
        raise ValueError('own_only and local_replicate are mutually exclusive')
    if initialize_wire and (own_only or local_replicate):
        raise ValueError('initialize_wire requires remote exchange')
    if (not isinstance(ranks,int) or not 1 <= ranks <= 128
            or not isinstance(width,int) or width <= 0 or width%128):
        raise ValueError('invalid S5 histogram exchange geometry')
    def exchange(source,out,local_sem,send,recv,ready,staging):
        def own_tile(i,_):
            section = pl.ds(i*128,128)
            load = pltpu.make_async_copy(source.at[:,section],staging,local_sem)
            load.start()
            load.wait()
            # Diagnostic initialization retains the remote path unchanged.
            for destination in range(ranks if local_replicate or initialize_wire else 1):
                store = pltpu.make_async_copy(staging,out.at[pl.ds(2*destination,2),section],local_sem)
                store.start()
                store.wait()
        lax.fori_loop(0,width//128,own_tile,None)
        for offset in range(1,1 if own_only or local_replicate else ranks):
            rank = lax.axis_index('core')
            right = lax.rem(rank+offset,jnp.int32(ranks))
            left = lax.rem(rank-offset+ranks,jnp.int32(ranks))
            pl.semaphore_signal(ready.at[offset-1],inc=1,device_id=(left,),
                                device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(ready.at[offset-1],1)
            transfer = pltpu.make_async_remote_copy(source,out.at[pl.ds(2*offset,2)],
                send_sem=send.at[offset-1],recv_sem=recv.at[offset-1],
                device_id=(right,),device_id_type=pl.DeviceIdType.MESH)
            transfer.start()
            transfer.wait_send()
            transfer.wait_recv()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    wire_call = pl.pallas_call(exchange,
        out_shape=jax.ShapeDtypeStruct((2 if own_only else 2*ranks,width),jnp.uint32),
        in_specs=(hbm,),out_specs=hbm,
        scratch_shapes=(pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((max(1,ranks-1),)),
            pltpu.SemaphoreType.DMA((max(1,ranks-1),)),
            pltpu.SemaphoreType.REGULAR((max(1,ranks-1),)),
            pltpu.VMEM((2,128),jnp.uint32)),
        interpret=interpret,name='beam_s5_histogram_exchange')
    def call(histogram):
        if histogram.shape != (2,width) or histogram.dtype != jnp.uint32:
            raise ValueError('local histogram must be uint32 low/high [2,width]')
        wire = wire_call(histogram)
        # Diagnostic-only exposure distinguishes transport from reduction.
        return wire if return_wire or own_only or local_replicate else pallas_sum_histogram_pairs(wire,interpret=interpret)
    return call
