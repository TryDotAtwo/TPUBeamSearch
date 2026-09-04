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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires exactly eight real TPU devices')

    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
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
        scope='one-hop eight-device Pallas RDMA correctness and diagnostic timing; not S5 ring completion',
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
