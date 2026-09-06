import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failure', [None, 'abort', 'partial'])
def test_isolated_groups_and_complete_case_gate(tmp_path, failure):
    from benchmarks.beam_final_bundle import run_bundle
    calls = []

    def runner(command, **kwargs):
        folder = Path(command[command.index('--output') + 1])
        calls.append(folder.name)
        materialize = folder.name == 'cuda_final'
        count = 6 if materialize else 16
        if failure == 'partial' and materialize:
            count -= 1
        data = {'all_exact' if materialize else 'exact': True,
                'cases': [{'name': str(i), 'exact': True} for i in range(count)]}
        (folder / f'{folder.name}.json').write_text(json.dumps(data))
        return SimpleNamespace(returncode=-6 if failure == 'abort' and materialize else 0)

    report = run_bundle(tmp_path, runner=runner)
    assert calls == ['cuda_final', 'final_exchange']
    assert report['all_exact'] == (failure is None)
    assert report['groups'][1]['exact']
    assert json.loads((tmp_path / 'final_bundle.json').read_text()) == report
