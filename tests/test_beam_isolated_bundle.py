"""A native compiler abort must not suppress independent diagnostics."""
import json
import subprocess

from benchmarks.beam_isolated_bundle import run_groups


def test_native_abort_preserves_partial_report_and_runs_next_group(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        destination = tmp_path / ('broken' if len(calls) == 1 else 'healthy')
        (destination / 'beam_primitives.json').write_text(json.dumps({
            'cases': [{'name': destination.name, 'status': 'started' if len(calls) == 1 else 'exact'}],
            'all_exact': len(calls) != 1,
        }))
        return subprocess.CompletedProcess(command, -6 if len(calls) == 1 else 0)

    report = run_groups(tmp_path, {'broken': ['a'], 'healthy': ['b']}, runner=runner)
    assert len(calls) == 2
    assert not report['all_exact']
    assert report['groups'][0]['returncode'] == -6
    assert report['groups'][0]['report']['cases'][0]['status'] == 'started'
    assert report['groups'][1]['report']['all_exact']
    assert json.loads((tmp_path / 'isolated_bundle.json').read_text()) == report


def test_exit_zero_without_results_is_not_success(tmp_path):
    result = run_groups(tmp_path, {'missing': ['x']},
        runner=lambda *a, **k: subprocess.CompletedProcess(a[0], 0))
    assert not result['all_exact']


def test_exact_gate_requires_every_requested_case(tmp_path):
    def runner(command, **kwargs):
        (tmp_path / 'group' / 'beam_primitives.json').write_text(json.dumps({
            'all_exact': True, 'cases': [{'name': 'a', 'status': 'exact'}]}))
        return subprocess.CompletedProcess(command, 0)

    assert run_groups(tmp_path, {'group': ['a']}, runner=runner)['all_exact']
    assert not run_groups(tmp_path, {'group': ['a', 'b']}, runner=runner)['all_exact']
