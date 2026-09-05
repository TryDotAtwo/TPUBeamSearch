import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('first_exact,returncode,expected_groups',[(False,1,1),(False,0,1),(True,0,2)])
def test_recovery_gate_does_not_promote_failed_full_collector(tmp_path,first_exact,returncode,expected_groups):
    assert importlib.util.find_spec('benchmarks.beam_collector_recovery_bundle') is not None
    from benchmarks.beam_collector_recovery_bundle import run_bundle
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        full = command[2] == 'benchmarks.beam_collector_full_probe'
        filename = 'collector_full.json' if full else 'stream3_collector.json'
        (folder/filename).write_text(json.dumps({'exact':first_exact if full else True}))
        return SimpleNamespace(returncode=returncode if full else 0)
    report = run_bundle(tmp_path,runner=runner)
    saved = json.loads((tmp_path/'recovery_bundle.json').read_text())
    assert saved == report
    assert len(report['groups']) == expected_groups
    assert report['all_exact'] == (first_exact and returncode == 0)
    assert (tmp_path/'integrated').exists() == (expected_groups == 2)


@pytest.mark.parametrize('artifact,returncode',[(None,-6),('{',1),('{"exact":true}',-6)])
def test_recovery_keeps_partial_failure_without_starting_integrated(tmp_path,artifact,returncode):
    from benchmarks.beam_collector_recovery_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        calls.append(command)
        folder = Path(command[command.index('--output')+1])
        if artifact is not None:
            (folder/'collector_full.json').write_text(artifact)
        kwargs['stdout'].write('partial native diagnostic\n')
        return SimpleNamespace(returncode=returncode)
    report = run_bundle(tmp_path,runner=runner)
    assert len(calls) == 1
    assert not report['all_exact']
    assert report['groups'][0]['returncode'] == returncode
    assert not (tmp_path/'integrated').exists()
    assert (tmp_path/'full/process.log').read_text() == 'partial native diagnostic\n'
    assert json.loads((tmp_path/'recovery_bundle.json').read_text()) == report
