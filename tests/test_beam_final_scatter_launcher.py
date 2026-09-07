from pathlib import Path
import subprocess


def test_launcher_leaves_fresh_output_for_scatter_coordinator(tmp_path, monkeypatch):
    from kaggle_beam_final_gate import run
    checkout = tmp_path/'checkout'
    output = tmp_path/'beam_final_scatter'
    monkeypatch.setattr(run, 'CHECKOUT', checkout)
    monkeypatch.setattr(run, 'OUTPUT', output)
    commands = []
    def external(command, **kwargs):
        commands.append(tuple(command))
        if 'benchmarks.beam_final_scatter_bundle' in command:
            assert not output.exists(), 'coordinator must own directory creation'
            assert kwargs['cwd'] == checkout
            assert kwargs['env']['JAX_ENABLE_X64'] == 'False'
            assert command[-2:] == ('--source-sha', run.COMMIT_SHA)
            output.mkdir()
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(run.subprocess, 'run', external)
    run.main()
    assert output.is_dir()
    assert commands[-1][2:4] == ('benchmarks.beam_final_scatter_bundle', '--output')
    assert commands[2] == ('git', 'checkout', '--detach', run.COMMIT_SHA)
