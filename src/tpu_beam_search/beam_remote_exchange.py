"""Snapshot-based variable-count RDMA, extracted without protocol changes.

Nonzero peers transfer full capacity; snapshots retain consumed payloads.
Not yet a direct-to-collector or count-proportional transport.
"""
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_variable_exchange_call(mesh, *, capacity=128):
    """Seven peer-offset epochs with a count preflight and bounded payload."""
    ranks = mesh.size
    epochs = ranks - 1

    def kernel(count_control_ref, payload_ref, count_ref, neutral_ref,
               payload_out, count_out, local_sems, count_local_sems,
               count_send_sems, count_recv_sems,
               payload_send_sems, payload_recv_sems, ready_sems, ack_sems,
               count_vmem):
        epoch = pl.program_id(0)
        slot = lax.rem(epoch, jnp.int32(2))
        offset = epoch + 1
        my_id = lax.axis_index('core')
        right = lax.rem(my_id + offset, jnp.int32(ranks))
        left = lax.rem(my_id - offset + ranks, jnp.int32(ranks))

        # A receiver releases exactly the source that targets it this epoch.
        pl.semaphore_signal(ready_sems.at[slot], inc=1, device_id=(left,),
                            device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(ready_sems.at[slot], 1)

        count_copy = pltpu.make_async_remote_copy(
            src_ref=count_ref.at[epoch], dst_ref=count_out.at[epoch],
            send_sem=count_send_sems.at[slot],
            recv_sem=count_recv_sems.at[slot], device_id=(right,),
            device_id_type=pl.DeviceIdType.MESH)
        count_copy.start()
        count_copy.wait_send()
        count_copy.wait_recv()
        stage_count = pltpu.make_async_copy(
            src_ref=count_out.at[epoch], dst_ref=count_vmem.at[slot],
            sem=count_local_sems.at[slot])
        stage_count.start()
        stage_count.wait()

        payload_copy = pltpu.make_async_remote_copy(
            src_ref=payload_ref.at[epoch], dst_ref=payload_out.at[slot],
            send_sem=payload_send_sems.at[slot],
            recv_sem=payload_recv_sems.at[slot], device_id=(right,),
            device_id_type=pl.DeviceIdType.MESH)
        # Predicate inputs must be readable on-chip. The local outgoing count
        # is scalar-prefetched, while the received HBM count is DMA-staged.
        send_nonzero = count_control_ref[epoch, 0] != 0
        recv_nonzero = count_vmem[slot, 0] != 0

        @pl.when(send_nonzero)
        def _send():
            payload_copy.start()
            payload_copy.wait_send()

        @pl.when(recv_nonzero)
        def _recv():
            payload_copy.wait_recv()
            consume = pltpu.make_async_copy(
                src_ref=payload_out.at[slot],
                dst_ref=payload_out.at[2 + epoch],
                sem=local_sems.at[slot])
            consume.start()
            consume.wait()

        @pl.when(~recv_nonzero)
        def _recv_zero():
            consume = pltpu.make_async_copy(
                src_ref=neutral_ref,
                dst_ref=payload_out.at[2 + epoch],
                sem=local_sems.at[slot])
            consume.start()
            consume.wait()

        pl.semaphore_signal(ack_sems.at[slot], inc=1, device_id=(left,),
                            device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(ack_sems.at[slot], 1)

    payload_shape = (epochs, 8, capacity)
    count_shape = (epochs, 128)
    output_payload_shape = (epochs + 2, 8, capacity)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 3,
        out_specs=(pl.BlockSpec(memory_space=pl.ANY),
                   pl.BlockSpec(memory_space=pl.ANY)),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.REGULAR((2,)),
            pltpu.SemaphoreType.REGULAR((2,)),
            pltpu.VMEM((2, 128), jnp.uint32),
        ),
        grid=(epochs,),
    )
    call = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(output_payload_shape, jnp.uint32),
                   jax.ShapeDtypeStruct(count_shape, jnp.uint32)),
        grid_spec=grid_spec,
        name='beam_stream3_variable_count_exchange',
    )
    return call


def make_exchange_collect_call(mesh,*,capacity=128):
    """Connect real snapshot RDMA to one aggregate remote collector admission.

    Local-owner collection remains a separate invocation. Every rank still
    executes all exchange epochs even if its collector already has fatal set.
    This functional composition needs physical validation and coordinated-stop
    integration; it does not promise aliases or concurrent S4 publication.
    """
    from .beam_receive_batch import pallas_collect_received
    exchange = make_variable_exchange_call(mesh,capacity=capacity)
    def rank_kernel(out):
        rank = lax.axis_index('core').astype(jnp.uint32)
        out[...] = (jnp.arange(128,dtype=jnp.int32)[None] == 0).astype(jnp.uint32)*rank
    rank_call = pl.pallas_call(rank_kernel,
        out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        out_specs=pl.BlockSpec((1,128)),grid=(),name='beam_receive_rank_control')
    def local_program(a,b,controls,wire,counts,neutral):
        snapshots,received_counts = exchange(counts,wire,counts,neutral)
        return pallas_collect_received(a,b,snapshots,controls,received_counts,rank_call())
    return local_program


def make_stream3_collect_call(mesh):
    """Bounded128 S3 threshold/dedup/owner/split -> local collect -> exchange.

    Metadata must already carry the correct parent/source/move for its payload.
    Does not restore S1/S2 ring payloads or implement coordinated fatal stop.
    Wire packing is deliberately limited to its physically exercised128 ABI.
    """
    from .beam_external_sort import pallas_external_stream3
    from .beam_stream3 import pallas_stream3_wire_slots
    from .beam_collector import pallas_collect
    exchange_collect = make_exchange_collect_call(mesh,capacity=128)
    def local_program(a,b,controls,words,payload,count,threshold,neutral):
        if words.shape != (8,128) or neutral.shape != (8,128):
            raise ValueError('integrated wire gate currently requires128 candidates')
        local,remote,local_count,send_count,send_offset = pallas_external_stream3(
            words,payload,count,threshold,local_rank=None,world_size=mesh.size)
        a,b,controls,_ = pallas_collect(a,b,local,controls,local_count)
        wire,wire_counts = pallas_stream3_wire_slots(
            remote,send_count,send_offset,local_rank=None,world_size=mesh.size)
        return exchange_collect(a,b,controls,wire,wire_counts,neutral)
    return local_program
