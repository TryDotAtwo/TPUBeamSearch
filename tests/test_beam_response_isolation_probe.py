import importlib.util
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('stage,want', [
    ('packing', [(8,32,128),(8,2,128)]),
    ('packing_guard', [(8,32,128),(8,2,128)]),
    ('packing_first_dma', [(8,32,128),(8,2,128)]),
    ('packing_second_dma', [(8,32,128),(8,2,128)]),
    ('packing_gather', [(8,32,128),(8,2,128)]),
    ('packing_row_copy', [(8,32,128),(8,2,128)]),
    ('packing_positions', [(8,32,128),(8,2,128)]),
    ('packing_clipped_positions', [(8,32,128),(8,2,128)]),
    ('packing_unmasked_gather', [(8,32,128),(8,2,128)]),
    ('packing_bounded_gather', [(8,32,128),(8,2,128)]),
    ('packing_rank2_gather', [(8,32,128),(8,2,128)]),
    ('packing_split_gather', [(8,32,128),(8,2,128)]),
    ('receive', [(32,1024),(2,128)]),
    ('planes_to_wire', [(1024,128)]),
])
def test_isolated_stage_traces_real_production_output_abi(stage,want):
    assert importlib.util.find_spec('benchmarks.beam_response_isolation_probe') is not None
    import jax
    from benchmarks.beam_response_isolation_probe import stage_call
    fn, inputs = stage_call(stage, SimpleNamespace(size=8), interpret=True)
    outputs = jax.eval_shape(fn, *inputs)
    assert [x.shape for x in outputs] == want
    assert str(outputs[0].dtype) == ('uint8' if stage=='planes_to_wire' else 'uint32')


def test_compile_runner_preserves_mlir_and_pending_state_on_compile_error(tmp_path):
    import json
    from benchmarks import beam_response_isolation_probe as probe
    assert hasattr(probe, 'save_compilation'), 'compile evidence writer missing'
    class Lowered:
        def as_text(self):
            return 'module before native compilation'
        def compile(self):
            assert (tmp_path/'lowered.mlir').read_text() == self.as_text()
            assert json.loads((tmp_path/'probe.json').read_text())['compiled'] is False
            raise RuntimeError('compiler rejection')
    with pytest.raises(RuntimeError, match='compiler rejection'):
        probe.save_compilation(Lowered(), tmp_path, {'stage':'packing'})
    assert not (tmp_path/'compiled.hlo.txt').exists()


def test_compile_evidence_writer_records_real_completed_compilation(tmp_path):
    import json
    import jax
    import jax.numpy as jnp
    from benchmarks.beam_response_isolation_probe import save_compilation
    lowered = jax.jit(lambda x: x+jnp.uint32(1)).lower(
        jax.ShapeDtypeStruct((128,), jnp.uint32))
    report = {'stage':'test_cpu_control', 'scope':'CPU writer test, not TPU evidence'}
    save_compilation(lowered, tmp_path, report)
    saved = json.loads((tmp_path/'probe.json').read_text())
    assert saved['compiled'] is True
    assert saved['status'] == 'compiled'
    assert (tmp_path/'compiled.hlo.txt').stat().st_size > 0


def test_cli_refuses_cpu_instead_of_reporting_tpu_success(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               PYTHONPATH=os.pathsep.join((str(root), str(root/'src'))))
    result = subprocess.run([sys.executable, '-m', 'benchmarks.beam_response_isolation_probe',
        '--stage', 'packing', '--output', str(tmp_path)], env=env,
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'requires eight physical TPU devices' in result.stderr
    assert not (tmp_path/'compiled.hlo.txt').exists()
