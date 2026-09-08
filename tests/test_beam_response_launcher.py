import json
import subprocess
import pytest


@pytest.mark.parametrize('rc',[0,-6])
def test_launcher_persists_child_failure_and_log(tmp_path,monkeypatch,rc):
    from kaggle_beam_response_gate import run
    monkeypatch.setattr(run,'CHECKOUT',tmp_path/'checkout')
    monkeypatch.setattr(run,'OUTPUT',tmp_path/'output')
    def external(command,**kwargs):
        if 'benchmarks.beam_response_epoch_probe' in command:
            assert not (run.OUTPUT/'probe').exists()
            assert kwargs['env']['JAX_ENABLE_X64']=='False'
            kwargs['stdout'].write('diagnostic output\n')
            return subprocess.CompletedProcess(command,rc)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(run.subprocess,'run',external)
    if rc:
        with pytest.raises(RuntimeError): run.main()
    else:
        run.main()
    report=json.loads((run.OUTPUT/'process.json').read_text())
    assert report['returncode']==rc and report['source_sha']==run.COMMIT_SHA
    assert (run.OUTPUT/'process.log').read_text()=='diagnostic output\n'
