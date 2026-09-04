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
    parser.add_argument('--mode', choices=('one-hop', 'slots'), default='one-hop')
    parser.add_argument('--zero-alternate', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires exactly eight real TPU devices')

    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    if args.mode == 'slots':
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
        scope=('two-slot multi-epoch eight-device Pallas RDMA diagnostic; not Stream3 exchange'
               if args.mode == 'slots' else
               'one-hop eight-device Pallas RDMA correctness and diagnostic timing; not S5 ring completion'),
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
