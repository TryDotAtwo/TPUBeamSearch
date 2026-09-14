import importlib.util
import json
import subprocess
import pytest


@pytest.mark.parametrize('rc', [0,1])
def test_isolation_launcher_runs_bundle_and_preserves_failure(tmp_path,monkeypatch,rc):
    assert importlib.util.find_spec('kaggle_beam_response_isolation') is not None
    from kaggle_beam_response_isolation import run
    monkeypatch.setattr(run,'CHECKOUT',tmp_path/'checkout')
    monkeypatch.setattr(run,'OUTPUT',tmp_path/'output')
    invoked=[]
    def external(command,**kwargs):
        if 'benchmarks.beam_response_isolation_bundle' in command:
            invoked.append(command)
            assert kwargs['env']['JAX_ENABLE_X64']=='False'
            kwargs['stdout'].write('all stage diagnostics retained\n')
            return subprocess.CompletedProcess(command,rc)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(run.subprocess,'run',external)
    if rc:
        with pytest.raises(RuntimeError):
            run.main()
    else:
        run.main()
    assert len(invoked)==1
    assert json.loads((run.OUTPUT/'process.json').read_text())['returncode']==rc
    assert 'all stage diagnostics' in (run.OUTPUT/'process.log').read_text()
