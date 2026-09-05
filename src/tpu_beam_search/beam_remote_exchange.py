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
