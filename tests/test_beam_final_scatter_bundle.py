import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.beam_final_scatter_fixture import make_case

SHA = 'a' * 40


def report_for(mode):
    cases = [make_case(n) for n in (0, 1, 127, 128, 129)]
    if mode == 'scatter':
        cases += [make_case(129, failure=f) for f in ('count_overflow', 'target_overflow')]
    rows = []
    for case in cases:
        digest = hashlib.sha256(case['expected'].tobytes()).hexdigest()
        rows.append(dict(name=case['name'], exact=True, status='executed',
            input_sha256=hashlib.sha256(b''.join(x.tobytes() for x in case['inputs'])).hexdigest(),
            expected_sha256=digest, expected_error=case['expected_error'],
            mismatches=[0]*8, invalid=[case['expected_error']]*8,
            output_sha256=[digest]*8))
    return dict(mode=mode, exact=True, source_sha=SHA, cases=rows,
                jax='0.10.2', jaxlib='0.10.2', libtpu='0.0.42.1',
                devices=[dict(id=i, kind='TPU v5 lite') for i in range(8)])


def test_accepts_complete_eight_device_report():
    from benchmarks.beam_final_scatter_bundle import validate_report
    assert validate_report(report_for('scatter'), 'scatter', SHA) == []


@pytest.mark.parametrize('mutation', [
    'missing', 'duplicate', 'extra', 'mode', 'source', 'pending', 'mismatch',
    'boolean_counter', 'short_hashes', 'forged_expected', 'input_hash',
    'error_count', 'device_duplicate', 'cpu', 'runtime', 'exact', 'malformed',
])
def test_rejects_false_positive_report(mutation):
    from benchmarks.beam_final_scatter_bundle import validate_report
    value = report_for('scatter')
    row = value['cases'][0]
    if mutation == 'missing': value['cases'].pop()
    elif mutation == 'duplicate': value['cases'][-1] = copy.deepcopy(row)
    elif mutation == 'extra': value['cases'].append(copy.deepcopy(row))
    elif mutation == 'mode': value['mode'] = 'integrated'
    elif mutation == 'source': value['source_sha'] = 'b'*40
    elif mutation == 'pending': row['status'] = 'compiling'
    elif mutation == 'mismatch': row['mismatches'][0] = 1
    elif mutation == 'boolean_counter': row['mismatches'][0] = False
    elif mutation == 'short_hashes': row['output_sha256'].pop()
    elif mutation == 'forged_expected':
        row['expected_sha256'] = 'b'*64
        row['output_sha256'] = ['b'*64]*8
    elif mutation == 'input_hash': row['input_sha256'] = 'b'*64
    elif mutation == 'error_count': value['cases'][-1]['invalid'] = [0]*8
    elif mutation == 'device_duplicate': value['devices'][-1]['id'] = 0
    elif mutation == 'cpu': value['devices'][0]['kind'] = 'cpu'
    elif mutation == 'runtime': value['libtpu'] = ''
    elif mutation == 'exact': value['exact'] = False
    elif mutation == 'malformed': value = []
    assert validate_report(value, 'scatter', SHA)


def test_abort_retains_partial_report_and_runs_other_mode(tmp_path):
    from benchmarks.beam_final_scatter_bundle import run_bundle
    def runner(command, **kwargs):
        mode = command[command.index('--mode')+1]
        folder = Path(command[-1])
        pending = json.loads((folder.parent/'scatter_bundle.json').read_text())
        assert pending['cases'][-1]['status'] == 'running'
        value = report_for(mode)
        if mode == 'scatter':
            value['exact'] = False
            value['cases'] = value['cases'][:1]
            value['cases'][0]['status'] = 'compiling'
        (folder/'probe.json').write_text(json.dumps(value))
        kwargs['stdout'].write('native abort' if mode == 'scatter' else 'done')
        return subprocess.CompletedProcess(command, -6 if mode == 'scatter' else 0)
    result = run_bundle(tmp_path/'run', source_sha=SHA, runner=runner)
    assert not result['all_exact']
    assert [r['returncode'] for r in result['cases']] == [-6, 0]
    assert result['cases'][1]['exact'] is True
    assert result['cases'][0]['report']['cases'][0]['status'] == 'compiling'
    assert (tmp_path/'run/scatter/process.log').read_text() == 'native abort'


def test_refuses_existing_destination(tmp_path):
    from benchmarks.beam_final_scatter_bundle import run_bundle
    with pytest.raises(FileExistsError):
        run_bundle(tmp_path, source_sha=SHA)


@pytest.mark.parametrize('different_runtime', [False, True])
def test_bundle_requires_matching_runtime(tmp_path, different_runtime):
    from benchmarks.beam_final_scatter_bundle import run_bundle
    def runner(command, **kwargs):
        mode = command[command.index('--mode')+1]
        value = report_for(mode)
        if different_runtime and mode == 'integrated': value['jax'] = 'other'
        (Path(command[-1])/'probe.json').write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, 0)
    result = run_bundle(tmp_path/'run', source_sha=SHA, runner=runner)
    assert result['all_exact'] is (not different_runtime)
