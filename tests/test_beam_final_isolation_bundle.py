import json
from pathlib import Path
from types import SimpleNamespace


def test_native_abort_does_not_skip_later_isolation_cases(tmp_path):
    from benchmarks.beam_final_isolation_bundle import run_bundle, MODES
    calls = []
    def runner(command, **kwargs):
        mode = command[command.index('--mode')+1]
        folder = Path(command[command.index('--output')+1])
        calls.append(mode)
        if mode != MODES[0]:
            (folder/'probe.json').write_text(json.dumps({'mode':mode,'exact':True}))
        return SimpleNamespace(returncode=-6 if mode == MODES[0] else 0)
    report = run_bundle(tmp_path,runner=runner)
    assert calls == list(MODES)
    assert not report['all_exact']
    assert report['cases'][0]['returncode'] == -6
    assert all(row['exact'] for row in report['cases'][1:])
    assert json.loads((tmp_path/'isolation_bundle.json').read_text()) == report
