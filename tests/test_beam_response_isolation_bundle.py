import importlib.util
import json
import pytest
from pathlib import Path
from types import SimpleNamespace


def test_abort_and_missing_compile_evidence_do_not_hide_remaining_stages(tmp_path):
    name = 'benchmarks.beam_response_isolation_bundle'
    assert importlib.util.find_spec(name) is not None, 'response isolation coordinator missing'
    from benchmarks.beam_response_isolation_bundle import run_bundle
    visited = []

    def runner(command, **kwargs):
        stage = command[command.index('--stage') + 1]
        folder = Path(command[command.index('--output') + 1])
        visited.append(stage)
        pending = json.loads((tmp_path/'response_isolation.json').read_text())
        assert pending['cases'][-1]['stage'] == stage
        assert pending['cases'][-1]['status'] == 'running'
        kwargs['stdout'].write('diagnostic ' + stage)
        # External TPU subprocess is the only substituted dependency.
        (folder/'probe.json').write_text(json.dumps({'stage': stage, 'compiled': True}))
        if stage != 'receive':
            (folder/'lowered.mlir').write_text('module {}')
            (folder/'compiled.hlo.txt').write_text('HloModule diagnostic')
        return SimpleNamespace(returncode=-6 if stage == 'packing' else 0)

    report = run_bundle(tmp_path, runner=runner)
    assert visited == ['packing', 'packing_control', 'packing_selection',
        'packing_guard', 'packing_first_dma', 'packing_second_dma', 'packing_gather',
        'packing_row_copy', 'packing_positions', 'packing_clipped_positions', 'packing_unmasked_gather',
        'exchange', 'receive', 'planes_to_wire', 'composition']
    assert {r['stage'] for r in report['cases'] if not r['compiled']} == {'packing','receive'}
    assert report['cases'][0]['returncode'] == -6
    assert not report['all_compiled']
    assert 'all_exact' not in report
    assert json.loads((tmp_path/'response_isolation.json').read_text()) == report
    assert (tmp_path/'composition'/'process.log').read_text() == 'diagnostic composition'


@pytest.mark.parametrize('contents',[None,'{"compiled":','[]','null'])
def test_missing_or_partial_nested_report_preserves_later_diagnostics(tmp_path,contents):
    from benchmarks.beam_response_isolation_bundle import run_bundle,STAGES
    def runner(command,**kwargs):
        stage=command[command.index('--stage')+1]
        folder=Path(command[command.index('--output')+1])
        if stage == 'packing':
            if contents is not None:
                (folder/'probe.json').write_text(contents)
        else:
            (folder/'probe.json').write_text(json.dumps(dict(stage=stage,compiled=True)))
        (folder/'lowered.mlir').write_text('diagnostic fixture')
        (folder/'compiled.hlo.txt').write_text('diagnostic fixture')
        return SimpleNamespace(returncode=0)
    report=run_bundle(tmp_path,runner=runner)
    assert len(report['cases'])==len(STAGES)
    assert not report['cases'][0]['compiled']
    assert 'report_error' in report['cases'][0]
    assert all(row['compiled'] for row in report['cases'][1:])
    assert not report['all_compiled']
    assert json.loads((tmp_path/'response_isolation.json').read_text())==report
