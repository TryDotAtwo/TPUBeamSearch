import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def test_bundle_continues_after_failed_child_and_never_promotes_missing_result(tmp_path):
    assert importlib.util.find_spec('benchmarks.beam_collector_bundle') is not None
    from benchmarks.beam_collector_bundle import run_bundle
    calls = []
    def runner(command,**kwargs):
        folder = Path(command[command.index('--output')+1])
        calls.append(folder.name)
        if len(calls) == 1:
            return SimpleNamespace(returncode=-6)
        if folder.name == 'group':
            (folder/'collector.json').write_text(json.dumps({'cases':[{'exact':True}]}))
        # full child returns 0 without JSON: must not be accepted.
        return SimpleNamespace(returncode=0)
    result = run_bundle(tmp_path,runner=runner)
    assert calls == ['single','group','full']
    assert result['all_exact'] is False
    assert [x['exact'] for x in result['groups']] == [False,True,False]
    assert result['groups'][0]['returncode'] == -6
