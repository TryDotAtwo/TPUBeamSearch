"""Physical eight-TPU Pallas remote-DMA ring probe.

This is the first transport gate only: one push to the right neighbor with
separate send/receive waits. Slot reuse and zero-count epochs are later gates.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def evaluate_right_permute(actual, source):
    actual = np.asarray(actual)
    source = np.asarray(source)
    expected = np.roll(source, 1, axis=0)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(exact=False, mismatched_elements=None,
                    structure_mismatch=True)
    mismatches = int(np.count_nonzero(actual != expected))
    return dict(
        exact=mismatches == 0,
        mismatched_elements=mismatches,
        structure_mismatch=False,
        input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
    )


def evaluate_epoch_ring(actual, source, *, active_epochs):
    actual = np.asarray(actual)
    source = np.asarray(source)
    if source.ndim < 2 or len(active_epochs) != source.shape[1]:
        raise ValueError('active_epochs must match the epoch axis')
    expected = np.roll(source, 1, axis=0)
    for epoch, active in enumerate(active_epochs):
        if not active:
            expected[:, epoch] = 0
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(exact=False, mismatched_elements=None,
                    structure_mismatch=True)
    mismatches = int(np.count_nonzero(actual != expected))
    return dict(exact=mismatches == 0, mismatched_elements=mismatches,
                structure_mismatch=False,
                output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest())


def evaluate_variable_exchange(actual, actual_counts, source, send_counts):
    actual = np.asarray(actual)
    actual_counts = np.asarray(actual_counts)
    source = np.asarray(source)
    send_counts = np.asarray(send_counts)
    if source.ndim != 4 or source.shape[1] != source.shape[0] - 1:
        raise ValueError('source must be [rank, rank-1, plane, capacity]')
    if send_counts.shape != source.shape[:2]:
        raise ValueError('send_counts must match rank and peer-offset axes')
    expected = np.empty_like(source)
    expected_counts = np.empty_like(send_counts)
    ranks = source.shape[0]
    for destination in range(ranks):
        for offset_index in range(ranks - 1):
            sender = (destination - offset_index - 1) % ranks
            expected[destination, offset_index] = source[sender, offset_index]
            expected_counts[destination, offset_index] = send_counts[sender, offset_index]
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual_counts.shape != expected_counts.shape
            or actual_counts.dtype != expected_counts.dtype):
        return dict(exact=False, mismatched_elements=None,
                    structure_mismatch=True)
    mismatches = (int(np.count_nonzero(actual != expected))
                  + int(np.count_nonzero(actual_counts != expected_counts)))
    return dict(
        exact=mismatches == 0,
        mismatched_elements=mismatches,
        structure_mismatch=False,
        output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        count_sha256=hashlib.sha256(actual_counts.tobytes()).hexdigest(),
        expected_count_sha256=hashlib.sha256(expected_counts.tobytes()).hexdigest(),
    )


def make_right_permute(mesh):
    device_count = mesh.size

    def kernel(input_ref, output_ref, send_sem, recv_sem):
        my_id = lax.axis_index('core')
        right_neighbor = lax.rem(my_id + 1, device_count)
        copy = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=output_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(right_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copy.wait_send()
        copy.wait_recv()

    local_shape = (8, 128)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=([pltpu.SemaphoreType.DMA] * 2),
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(local_shape, jnp.uint32),
        grid_spec=grid_spec,
        name='beam_rdma_right_permute',
    )
    partition = jax.sharding.PartitionSpec(None, 'core')
    return jax.jit(jax.shard_map(
        call, mesh=mesh, in_specs=partition, out_specs=partition,
        check_vma=False,
    ))


def make_epoch_ring(mesh, *, epochs, zero_alternate):
    """Two-slot ring with readiness and post-consumption acknowledgement."""
    device_count = mesh.size

    def kernel(input_ref, output_ref, local_sems, send_sems, recv_sems,
               ready_sems, ack_sems):
        epoch = pl.program_id(0)
        slot = lax.rem(epoch, 2)
        my_id = lax.axis_index('core')
        right_neighbor = lax.rem(my_id + 1, device_count)
        left_neighbor = lax.rem(my_id - 1 + device_count, device_count)
        active = (epoch & 1) == 0 if zero_alternate else jnp.bool_(True)

        @pl.when(active)
        def _active():
            # Receiver advertises destination capacity to its left sender.
            pl.semaphore_signal(
                ready_sems.at[slot], inc=1, device_id=(left_neighbor,),
                device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(ready_sems.at[slot], 1)
            remote = pltpu.make_async_remote_copy(
                src_ref=input_ref.at[epoch],
                dst_ref=output_ref.at[slot],
                send_sem=send_sems.at[slot],
                recv_sem=recv_sems.at[slot],
                device_id=(right_neighbor,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            remote.start()
            remote.wait_send()
            remote.wait_recv()
            consume = pltpu.make_async_copy(
                src_ref=output_ref.at[slot],
                dst_ref=output_ref.at[2 + epoch],
                sem=local_sems.at[slot],
            )
            consume.start()
            consume.wait()
            # The receiver consumed its slot and releases the left sender.
            pl.semaphore_signal(
                ack_sems.at[slot], inc=1, device_id=(left_neighbor,),
                device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(ack_sems.at[slot], 1)

        @pl.when(~active)
        def _inactive():
            # Input for inactive epochs is explicitly zero. This local copy
            # records participation without starting/waiting on remote DMA.
            consume = pltpu.make_async_copy(
                src_ref=input_ref.at[epoch],
                dst_ref=output_ref.at[2 + epoch],
                sem=local_sems.at[slot],
            )
            consume.start()
            consume.wait()

    local_input_shape = (epochs, 8, 128)
    local_output_shape = (epochs + 2, 8, 128)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.REGULAR((2,)),
            pltpu.SemaphoreType.REGULAR((2,)),
        ),
        grid=(epochs,),
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(local_output_shape, jnp.uint32),
        grid_spec=grid_spec,
        name='beam_rdma_two_slot_epoch_ring',
    )
    partition = jax.sharding.PartitionSpec(None, None, 'core')
    return jax.jit(jax.shard_map(
        call, mesh=mesh, in_specs=partition, out_specs=partition,
        check_vma=False,
    ))


def make_variable_exchange(mesh, *, capacity=128):
    """Seven peer-offset epochs with a count preflight and bounded payload."""
    ranks = mesh.size
    epochs = ranks - 1

    def kernel(payload_ref, count_ref, neutral_ref, payload_out, count_out,
               local_sems, count_send_sems, count_recv_sems,
               payload_send_sems, payload_recv_sems, ready_sems, ack_sems):
        epoch = pl.program_id(0)
        slot = lax.rem(epoch, 2)
        offset = epoch + 1
        my_id = lax.axis_index('core')
        right = lax.rem(my_id + offset, ranks)
        left = lax.rem(my_id - offset + ranks, ranks)

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

        payload_copy = pltpu.make_async_remote_copy(
            src_ref=payload_ref.at[epoch], dst_ref=payload_out.at[slot],
            send_sem=payload_send_sems.at[slot],
            recv_sem=payload_recv_sems.at[slot], device_id=(right,),
            device_id_type=pl.DeviceIdType.MESH)
        send_nonzero = count_ref[epoch, 0] != 0
        recv_nonzero = count_out[epoch, 0] != 0

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
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 3,
        out_specs=(pl.BlockSpec(memory_space=pl.ANY),
                   pl.BlockSpec(memory_space=pl.ANY)),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.REGULAR((2,)),
            pltpu.SemaphoreType.REGULAR((2,)),
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
    payload_partition = jax.sharding.PartitionSpec(None, None, 'core')
    count_partition = jax.sharding.PartitionSpec(None, 'core')
    neutral_partition = jax.sharding.PartitionSpec(None, 'core')
    return jax.jit(jax.shard_map(
        call, mesh=mesh,
        in_specs=(payload_partition, count_partition, neutral_partition),
        out_specs=(payload_partition, count_partition), check_vma=False))


def build_variable_exchange_inputs(ranks=8, capacity=128):
    epochs = ranks - 1
    neutral = np.zeros((8, capacity), np.uint32)
    neutral[6] = np.uint32(0xffffffff)
    payload = np.broadcast_to(neutral, (ranks, epochs, 8, capacity)).copy()
    counts = np.zeros((ranks, epochs, 128), np.uint32)
    for rank in range(ranks):
        for epoch in range(epochs):
            count = (rank * 3 + (epoch + 1) * 5) % 6
            if rank == 0 and epoch == epochs - 1:
                count = capacity
            counts[rank, epoch, 0] = count
            for index in range(count):
                payload[rank, epoch, :, index] = np.array([
                    rank, epoch + 1, index, rank * 1000 + epoch * 128 + index,
                    index * 17, rank * 31, index + epoch, (rank << 16) | index,
                ], np.uint32)
    return payload, counts, neutral


def run_variable_exchange(args, devices, mesh):
    ranks, capacity = len(devices), 128
    payload, counts, neutral = build_variable_exchange_inputs(ranks, capacity)
    payload_global = payload.transpose(1, 2, 0, 3).reshape(ranks - 1, 8, ranks * capacity)
    counts_global = counts.transpose(1, 0, 2).reshape(ranks - 1, ranks * 128)
    neutral_global = np.tile(neutral, (1, ranks))
    payload_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, None, 'core'))
    count_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, 'core'))
    placed = (jax.device_put(payload_global, payload_sharding),
              jax.device_put(counts_global, count_sharding),
              jax.device_put(neutral_global, count_sharding))
    fn = make_variable_exchange(mesh, capacity=capacity)
    started = time.perf_counter()
    executable = fn.lower(*placed).compile()
    compile_seconds = time.perf_counter() - started
    (args.output / 'variable_exchange.hlo.txt').write_text(
        executable.as_text(), encoding='utf-8')
    payload_result, count_result = jax.block_until_ready(executable(*placed))
    payload_global_out = np.asarray(payload_result)[2:]
    actual = payload_global_out.reshape(ranks - 1, 8, ranks, capacity).transpose(2, 0, 1, 3)
    actual_counts = np.asarray(count_result).reshape(ranks - 1, ranks, 128).transpose(1, 0, 2)[:, :, 0]
    result = evaluate_variable_exchange(actual, actual_counts, payload, counts[:, :, 0])
    result.update(ranks=ranks, epochs=ranks - 1, capacity=capacity,
                  send_counts=counts[:, :, 0].tolist(),
                  compile_seconds=compile_seconds)
    return result, executable, placed


def run_epoch_ring(args, devices, mesh):
    epochs = 4
    partition = jax.sharding.PartitionSpec(None, None, 'core')
    sharding = jax.sharding.NamedSharding(mesh, partition)
    per_device = np.arange(
        8 * epochs * 8 * 128, dtype=np.uint32).reshape(8, epochs, 8, 128)
    per_device ^= np.arange(8, dtype=np.uint32)[:, None, None, None] * np.uint32(0x9e3779b9)
    if args.zero_alternate:
        per_device[:, 1::2] = 0
    global_input = per_device.transpose(1, 2, 0, 3).reshape(epochs, 8, 8 * 128)
    placed = jax.device_put(global_input, sharding)
    fn = make_epoch_ring(mesh, epochs=epochs, zero_alternate=args.zero_alternate)
    started = time.perf_counter()
    executable = fn.lower(placed).compile()
    compile_seconds = time.perf_counter() - started
    (args.output / 'epoch_ring.hlo.txt').write_text(executable.as_text(), encoding='utf-8')
    actual_global = np.asarray(jax.block_until_ready(executable(placed)))[2:]
    actual = actual_global.reshape(epochs, 8, 8, 128).transpose(2, 0, 1, 3)
    active_epochs = tuple(not args.zero_alternate or epoch % 2 == 0
                          for epoch in range(epochs))
    result = evaluate_epoch_ring(actual, per_device, active_epochs=active_epochs)
    result.update(epochs=epochs, active_epochs=active_epochs,
                  slot_count=2, compile_seconds=compile_seconds)
    return result, executable, placed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('one-hop', 'slots', 'variable'), default='one-hop')
    parser.add_argument('--zero-alternate', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires exactly eight real TPU devices')

    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    if args.mode == 'variable':
        result, executable, placed = run_variable_exchange(args, devices, mesh)
        compile_seconds = result['compile_seconds']
    elif args.mode == 'slots':
        result, executable, placed = run_epoch_ring(args, devices, mesh)
        compile_seconds = result['compile_seconds']
    else:
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, 'core'))
        per_device = np.arange(8 * 8 * 128, dtype=np.uint32).reshape(8, 8, 128)
        per_device ^= np.arange(8, dtype=np.uint32)[:, None, None] * np.uint32(0x9e3779b9)
        global_input = per_device.transpose(1, 0, 2).reshape(8, 8 * 128)
        placed = jax.device_put(global_input, sharding)
        fn = make_right_permute(mesh)
        started = time.perf_counter()
        executable = fn.lower(placed).compile()
        compile_seconds = time.perf_counter() - started
        (args.output / 'right_permute.hlo.txt').write_text(
            executable.as_text(), encoding='utf-8')
        actual_global = np.asarray(jax.block_until_ready(executable(placed)))
        actual = actual_global.reshape(8, 8, 128).transpose(1, 0, 2)
        result = evaluate_right_permute(actual, per_device)

    for _ in range(3):
        jax.block_until_ready(executable(placed))
    samples = []
    for _ in range(21):
        start = time.perf_counter_ns()
        jax.block_until_ready(executable(placed))
        samples.append((time.perf_counter_ns() - start) / 1e6)
    result.update(
        scope=('bounded Stream3 variable-count RDMA diagnostic'
               if args.mode == 'variable' else
               ('two-slot multi-epoch eight-device Pallas RDMA diagnostic; not Stream3 exchange'
                if args.mode == 'slots' else
                'one-hop eight-device Pallas RDMA correctness and diagnostic timing; not S5 ring completion')),
        source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
        libtpu=importlib.metadata.version('libtpu'),
        devices=[dict(id=d.id, kind=d.device_kind, process_index=d.process_index)
                 for d in devices],
        compile_seconds=compile_seconds,
        timing=dict(warmup=3, repeats=21, samples_ms=samples,
                    median_ms=float(np.median(samples)),
                    p10_ms=float(np.percentile(samples, 10)),
                    p90_ms=float(np.percentile(samples, 90))),
    )
    (args.output / 'beam_rdma_ring_probe.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if not result['exact']:
        raise RuntimeError('right-permute RDMA mismatch')


if __name__ == '__main__':
    main()
