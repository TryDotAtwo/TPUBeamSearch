"""Compile-only isolation of unchanged response epoch production stages."""
import jax
import jax.numpy as jnp
from tpu_beam_search.beam_final_chunk import pallas_pack_final_chunk
from tpu_beam_search.beam_final_exchange import make_final_chunk_exchange
from tpu_beam_search.beam_final_receive import pallas_compact_final_received
from tpu_beam_search.beam_final_transport import pallas_planes_to_wire
from tpu_beam_search.beam_final_response_chunk import make_final_response_chunk_call


def stage_call(stage, mesh, *, interpret=False):
    """Local argument shapes match V4; every input remains a runtime argument."""
    def shape(*dims):
        return jax.ShapeDtypeStruct(dims, jnp.uint32)
    ranks = mesh.size
    if stage in ('packing_control','packing_selection'):
        from .beam_packing_control_probe import make_probe
        return make_probe(selection=stage=='packing_selection',world_size=ranks,interpret=interpret), (
            shape(32,2048),shape(3,128),shape(1,128),shape(1))
    if stage == 'packing':
        def call(payload, intervals, error, index):
            return pallas_pack_final_chunk(payload, intervals, index,
                world_size=ranks, prior_error=error, interpret=interpret)
        return call, (shape(32,2048), shape(3,128), shape(1,128), shape(1))
    if stage == 'exchange':
        return make_final_chunk_exchange(mesh, planes=32, interpret=interpret), (
            shape(ranks,32,128), shape(ranks,2,128))
    if stage == 'receive':
        def call(snapshots, counts, error):
            return pallas_compact_final_received(snapshots, counts, error, interpret=interpret)
        return call, (shape(ranks,32,128), shape(ranks,1,128), shape(1,128))
    if stage == 'planes_to_wire':
        def call(planes):
            return (pallas_planes_to_wire(planes, interpret=interpret),)
        return call, (shape(32, 1 << (ranks*128-1).bit_length()),)
    if stage == 'composition':
        return make_final_response_chunk_call(mesh, wire_width=128, interpret=interpret), (
            shape(35,2048), shape(3,128), shape(1,128), shape(1))
    raise ValueError('unknown response isolation stage')


def save_compilation(lowered, output, report):
    import json
    report.update(compiled=False, status='compiling')
    (output/'lowered.mlir').write_text(lowered.as_text())
    (output/'probe.json').write_text(json.dumps(report, indent=2))
    executable = lowered.compile()
    (output/'compiled.hlo.txt').write_text(executable.as_text())
    report.update(compiled=True, status='compiled')
    (output/'probe.json').write_text(json.dumps(report, indent=2))


def main():
    import argparse
    import importlib.metadata
    import json
    from pathlib import Path
    import subprocess
    import numpy as np
    from .beam_response_isolation_bundle import STAGES
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', False)
    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != 'tpu' for d in devices):
        raise RuntimeError('requires eight physical TPU devices')
    mesh = jax.sharding.Mesh(np.asarray(devices), ('core',))
    call, local_inputs = stage_call(args.stage, mesh)
    p = jax.sharding.PartitionSpec
    # The V4 epoch index is replicated, all other inputs are per-rank.
    replicated_index = args.stage in ('packing', 'packing_control', 'packing_selection', 'composition')
    specs = tuple(p() if replicated_index and i == len(local_inputs)-1 else p('core')
                  for i in range(len(local_inputs)))
    global_inputs = tuple(jax.ShapeDtypeStruct(
        x.shape if spec == p() else (8,)+x.shape, x.dtype,
        sharding=jax.sharding.NamedSharding(mesh, spec))
        for x, spec in zip(local_inputs, specs))
    def local(*xs):
        values = tuple(x if spec == p() else x[0] for x,spec in zip(xs,specs))
        return tuple(x[None] for x in call(*values))
    fn = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=specs,
                             out_specs=p('core'), check_vma=False))
    report = dict(stage=args.stage, compiled=False, status='lowering',
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime={k:importlib.metadata.version(k) for k in ('jax','jaxlib','libtpu')},
        devices=[dict(id=d.id,kind=d.device_kind) for d in devices],
        input_abi=[dict(shape=x.shape,dtype=str(x.dtype),sharding=str(s))
                   for x,s in zip(global_inputs,specs)],
        scope='compile only: no execution, correctness or timings')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'probe.json').write_text(json.dumps(report, indent=2))
    save_compilation(fn.lower(*global_inputs), args.output, report)


if __name__ == '__main__':
    main()
