"""The private launcher must refuse a checkout other than its pinned source."""
from pathlib import Path

import pytest


def test_width_bridge_launcher_rejects_wrong_source_before_gate(monkeypatch):
    from kaggle_beam_state_width_bridge import run as launcher

    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))

    def fake_output(args, **kwargs):
        if tuple(args) == ("git", "rev-parse", "HEAD"):
            return "0" * 40
        if tuple(args) == ("git", "status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(launcher.subprocess, "check_output", fake_output)
    with pytest.raises(RuntimeError, match="pinned source"):
        launcher.main()
    assert any(command[:3] == ("git", "checkout", "--detach") for command in calls)
    assert not any("benchmarks.beam_state_width_bridge_execution" in command
                   for command in calls)


def test_width_bridge_launcher_runs_gate_from_clean_pinned_checkout(monkeypatch):
    from kaggle_beam_state_width_bridge import run as launcher

    calls = []
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda args, **kwargs: calls.append((tuple(args), kwargs)))

    def fake_output(args, **kwargs):
        if tuple(args) == ("git", "rev-parse", "HEAD"):
            return launcher.COMMIT_SHA
        if tuple(args) == ("git", "status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_output)
    launcher.main()
    benchmark = [entry for entry in calls if
                 "benchmarks.beam_state_width_bridge_execution" in entry[0]]
    assert len(benchmark) == 1
    command, options = benchmark[0]
    assert command[command.index("--output") + 1] == str(
        Path("/kaggle/working/state_width_bridge_execution"))
    assert options["cwd"] == launcher.CHECKOUT
    assert options["check"] is True
