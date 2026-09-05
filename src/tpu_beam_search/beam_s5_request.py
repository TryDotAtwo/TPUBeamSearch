"""Serialized peer-offset request MAX; physical multi-rank validation pending."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_s5_request_call(mesh,*,interpret=False):
    """Every rank calls, even request0; lane0 is the uint32 request flag.

    Each peer receives an immutable original request into a distinct offset
    slot. No early return, payload predicate, slot reuse or histogram transfer.
    This supplies only the common decision, not the full S5 epoch lifecycle.
    """
    ranks = mesh.size
    if not isinstance(ranks,int) or not 1 <= ranks <= 128:
        raise ValueError('request rank count must be in [1,128]')
    def exchange(source,out,local_sem,send,recv,ready,staging):
        load = pltpu.make_async_copy(source,staging,local_sem)
        load.start()
        load.wait()
        own = pltpu.make_async_copy(staging,out.at[pl.ds(0,1)],local_sem)
        own.start()
        own.wait()
        for offset in range(1,ranks):
            rank = lax.axis_index('core')
            right = lax.rem(rank+offset,jnp.int32(ranks))
            left = lax.rem(rank-offset+ranks,jnp.int32(ranks))
            pl.semaphore_signal(ready.at[offset-1],inc=1,device_id=(left,),
                                device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(ready.at[offset-1],1)
            transfer = pltpu.make_async_remote_copy(source,out.at[pl.ds(offset,1)],
                send_sem=send.at[offset-1],recv_sem=recv.at[offset-1],
                device_id=(right,),device_id_type=pl.DeviceIdType.MESH)
            transfer.start()
            transfer.wait_send()
            transfer.wait_recv()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    wire_call = pl.pallas_call(exchange,
        out_shape=jax.ShapeDtypeStruct((ranks,128),jnp.uint32),
        in_specs=(hbm,),out_specs=hbm,
        scratch_shapes=(pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((max(1,ranks-1),)),
            pltpu.SemaphoreType.DMA((max(1,ranks-1),)),
            pltpu.SemaphoreType.REGULAR((max(1,ranks-1),)),
            pltpu.VMEM((1,128),jnp.uint32)),
        interpret=interpret,name='beam_s5_request_exchange')
    def maximum(wire,out):
        # Mosaic lacks unsigned reductions. Flip the sign bit to preserve
        # uint32 ordering in signed int32, then invert after signed MAX.
        sign = jnp.uint32(0x80000000)
        signed = (wire[...] ^ sign).astype(jnp.int32)
        value = (jnp.max(signed,axis=0).astype(jnp.uint32) ^ sign)[None]
        out[...] = jnp.where(jnp.arange(128)[None] == 0,value,jnp.uint32(0))
    reduce_call = pl.pallas_call(maximum,
        out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        in_specs=(pl.BlockSpec((ranks,128)),),out_specs=pl.BlockSpec((1,128)),
        interpret=interpret,name='beam_s5_request_max')
    def call(request):
        if request.shape != (1,128) or request.dtype != jnp.uint32:
            raise ValueError('request must be uint32 [1,128]')
        return reduce_call(wire_call(request))
    return call
