import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_epoch_control_runs_composition_after_failed_combined(tmp_path):
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls = []
    def runner(command, **kwargs):
        folder = Path(command[command.index('--output')+1])
        calls.append(folder.name)
        if folder.name == 'epoch':
            assert '--explicit-hbm-output' in command
        (folder/f's5_{folder.name}.json').write_text(json.dumps({'exact':True}))
        return SimpleNamespace(returncode=1 if folder.name == 'hbm_combined' else 0)
    assert not run_bundle(tmp_path,runner=runner,epoch_control=True)['all_exact']
    assert calls == ['hbm_combined','epoch']


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


def test_recovery_isolates_four_s5_groups_without_repeating_s4(tmp_path):
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        calls.append(folder.name)
        (folder/f's5_{folder.name}.json').write_text(json.dumps({'exact':True}))
        return SimpleNamespace(returncode=1 if folder.name == 'wire' else 0)
    report = run_bundle(tmp_path,runner=runner,recovery=True)
    assert calls == ['request','wire','reduction','combined']
    assert not report['all_exact']


def test_transport_isolation_runs_own_and_wire_even_after_failure(tmp_path):
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        calls.append(folder.name)
        (folder/f's5_{folder.name}.json').write_text(json.dumps({'exact':False}))
        return SimpleNamespace(returncode=1)
    report = run_bundle(tmp_path,runner=runner,transport=True)
    assert calls == ['own','wire']
    assert not report['all_exact']


def test_layout_control_does_not_repeat_accepted_or_unchanged_groups(tmp_path):
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        calls.append(folder.name)
        (folder/'s5_replicate.json').write_text(json.dumps({'exact':True}))
        return SimpleNamespace(returncode=0)
    assert run_bundle(tmp_path,runner=runner,layout_control=True)['all_exact']
    assert calls == ['replicate']


def test_initialized_control_runs_only_new_case(tmp_path):
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        calls.append(folder.name)
        (folder/'s5_initialized.json').write_text(json.dumps({'exact':True}))
        return SimpleNamespace(returncode=0)
    assert run_bundle(tmp_path,runner=runner,initialized_control=True)['all_exact']
    assert calls == ['initialized']


def test_hbm_controls_continue_after_first_failure(tmp_path):
    from benchmarks.beam_s4_s5_bundle import run_bundle
    calls=[]
    def runner(command,**kwargs):
        folder=Path(command[command.index('--output')+1])
        calls.append(folder.name)
        (folder/f's5_{folder.name}.json').write_text(json.dumps({'exact':folder.name!='hbm'}))
        return SimpleNamespace(returncode=1 if folder.name=='hbm' else 0)
    assert not run_bundle(tmp_path,runner=runner,hbm_control=True)['all_exact']
    assert calls == ['hbm','hbm_initialized']
