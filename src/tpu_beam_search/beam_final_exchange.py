"""Serialized bounded final all-to-all snapshots; physical acceptance pending."""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from .beam_s5_request import make_s5_request_call


def make_final_chunk_exchange(mesh,*,planes,interpret=False):
    """Input is destination-major, output source-major, capacity128 per peer.

    All ranks first agree whether any count/error is invalid. On common error
    no payload exchange starts. Otherwise zero-count peers exchange control
    only. Distinct source snapshots remain alive after the call; this is not
    a direct-to-consumer ring or compute/DMA overlap implementation.
    """
    ranks=mesh.size
    if not isinstance(ranks,int) or not 1<=ranks<=128 or not isinstance(planes,int) or planes<=0:
        raise ValueError('invalid final exchange geometry')
    def preflight(c,out):
        bad=jnp.any(c[:,0,0]>128)|jnp.any(c[:,1,:]!=0)
        out[...] = jnp.where(jnp.arange(128)[None,:]==0,bad.astype(jnp.uint32),jnp.uint32(0))
    check=pl.pallas_call(preflight,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_exchange_preflight')
    agree=make_s5_request_call(mesh,interpret=interpret)
    def kernel(payload,controls,error,out,counts,local_sem,cs,cr,ps,pr,ready,ack,stage,count_stage):
        rank=lax.axis_index('core') if ranks>1 else jnp.int32(0)
        # Initialize snapshots locally before releasing any remote sender.
        stage[...] = jnp.zeros((planes,128),jnp.uint32)
        count_stage[...] = jnp.zeros((2,128),jnp.uint32)
        for source in range(ranks):
            clear=pltpu.make_async_copy(stage,out.at[source],local_sem)
            clear.start()
            clear.wait()
            clear_count=pltpu.make_async_copy(count_stage.at[pl.ds(0,1)],counts.at[source],local_sem)
            clear_count.start()
            clear_count.wait()
        @pl.when(error[0,0]==0)
        def exchange():
            own_count=controls[rank,0,0]
            count_stage[0,:]=jnp.where(jnp.arange(128)==0,own_count,jnp.uint32(0))
            write_count=pltpu.make_async_copy(count_stage.at[pl.ds(0,1)],counts.at[rank],local_sem)
            write_count.start()
            write_count.wait()
            @pl.when(own_count!=0)
            def own():
                load=pltpu.make_async_copy(payload.at[rank],stage,local_sem)
                load.start()
                load.wait()
                store=pltpu.make_async_copy(stage,out.at[rank],local_sem)
                store.start()
                store.wait()
            for offset in range(1,ranks):
                right=lax.rem(rank+offset,jnp.int32(ranks))
                left=lax.rem(rank-offset+ranks,jnp.int32(ranks))
                slot=(offset-1)%2
                pl.semaphore_signal(ready.at[slot],1,device_id=(left,),device_id_type=pl.DeviceIdType.MESH)
                pl.semaphore_wait(ready.at[slot],1)
                outgoing=controls[right,0,0]
                count_stage[0,:]=jnp.where(jnp.arange(128)==0,outgoing,jnp.uint32(0))
                transfer_count=pltpu.make_async_remote_copy(count_stage.at[pl.ds(0,1)],counts.at[rank],
                    send_sem=cs.at[slot],recv_sem=cr.at[slot],device_id=(right,),device_id_type=pl.DeviceIdType.MESH)
                transfer_count.start()
                transfer_count.wait_send()
                transfer_count.wait_recv()
                read_count=pltpu.make_async_copy(counts.at[left],count_stage.at[pl.ds(1,1)],local_sem)
                read_count.start()
                read_count.wait()
                transfer=pltpu.make_async_remote_copy(payload.at[right],out.at[rank],
                    send_sem=ps.at[slot],recv_sem=pr.at[slot],device_id=(right,),device_id_type=pl.DeviceIdType.MESH)
                @pl.when(outgoing!=0)
                def send():
                    transfer.start()
                    transfer.wait_send()
                @pl.when(count_stage[1,0]!=0)
                def receive():
                    transfer.wait_recv()
                # Snapshot is retained, not overwritten by subsequent offsets.
                pl.semaphore_signal(ack.at[slot],1,device_id=(left,),device_id_type=pl.DeviceIdType.MESH)
                pl.semaphore_wait(ack.at[slot],1)
    hbm=pl.BlockSpec(memory_space=pltpu.HBM)
    wire=pl.pallas_call(kernel,
        out_shape=(pltpu.HBM((ranks,planes,128),jnp.uint32),pltpu.HBM((ranks,1,128),jnp.uint32)),
        in_specs=(hbm,pl.BlockSpec((ranks,2,128)),pl.BlockSpec((1,128))),out_specs=(hbm,hbm),
        scratch_shapes=(pltpu.SemaphoreType.DMA,*[pltpu.SemaphoreType.DMA((2,)) for _ in range(4)],
            pltpu.SemaphoreType.REGULAR((2,)),pltpu.SemaphoreType.REGULAR((2,)),
            pltpu.VMEM((planes,128),jnp.uint32),pltpu.VMEM((2,128),jnp.uint32)),
        interpret=interpret,name='beam_final_chunk_exchange')
    def call(payload,controls):
        if payload.shape!=(ranks,planes,128) or controls.shape!=(ranks,2,128) or any(x.dtype!=jnp.uint32 for x in (payload,controls)):
            raise ValueError('invalid final chunk exchange ABI')
        error=agree(check(controls))
        received,counts=wire(payload,controls,error)
        return received,counts,error
    return call
