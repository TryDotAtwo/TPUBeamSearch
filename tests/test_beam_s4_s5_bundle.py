import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('failure',[None,'s4','request'])
def test_bundle_preserves_failed_process_and_runs_remaining_independent_groups(tmp_path,failure):
    assert importlib.util.find_spec('benchmarks.beam_s4_s5_bundle') is not None
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        name = folder.name
        calls.append(name)
        filename = {'s4':'s4_reserved.json','request':'s5_request.json','histogram':'s5_histogram.json'}[name]
        # An exact partial report cannot override a failing process status.
        (folder/filename).write_text(json.dumps({'exact':True}))
        return SimpleNamespace(returncode=-6 if name == failure else 0)
    report = run_bundle(tmp_path,runner=runner)
    assert calls == ['s4','request','histogram']
    assert report['all_exact'] == (failure is None)
    assert [r['exact'] for r in report['groups']] == [n != failure for n in calls]
    assert json.loads((tmp_path/'s4_s5_bundle.json').read_text()) == report
