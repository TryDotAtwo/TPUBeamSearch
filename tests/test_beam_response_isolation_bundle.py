import importlib.util
import json
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
    assert visited == ['packing', 'packing_control', 'packing_selection', 'exchange', 'receive', 'planes_to_wire', 'composition']
    assert [r['compiled'] for r in report['cases']] == [False, True, True, True, False, True, True]
    assert report['cases'][0]['returncode'] == -6
    assert not report['all_compiled']
    assert 'all_exact' not in report
    assert json.loads((tmp_path/'response_isolation.json').read_text()) == report
    assert (tmp_path/'composition'/'process.log').read_text() == 'diagnostic composition'
