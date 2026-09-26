import json
from pathlib import Path
from types import SimpleNamespace


def test_followup_stops_after_failed_packing_and_keeps_partial_report(tmp_path):
    from benchmarks.beam_response_followup_bundle import run_bundle

    calls = []

    def runner(command, **_):
        calls.append(command)
        folder = Path(command[-1])
        (folder / "probe.json").write_text(json.dumps({"stage": "packing", "compiled": False}))
        return SimpleNamespace(returncode=-6)

    report = run_bundle(tmp_path / "bundle", runner=runner)
    assert not report["all_exact"]
    assert [row["name"] for row in report["steps"]] == ["packing"]
    assert report["steps"][0]["returncode"] == -6
    assert len(calls) == 1
    assert json.loads((tmp_path / "bundle" / "followup.json").read_text()) == report


def test_followup_accepts_two_compiles_and_all_33_exact_epochs(tmp_path):
    from benchmarks.beam_response_followup_bundle import run_bundle

    calls = []

    def runner(command, **_):
        calls.append(command)
        folder = Path(command[-1])
        if "--stage" in command:
            stage = command[command.index("--stage") + 1]
            (folder / "probe.json").write_text(json.dumps({"stage": stage, "compiled": True}))
            (folder / "lowered.mlir").write_text("lowered")
            (folder / "compiled.hlo.txt").write_text("hlo")
        else:
            assert not folder.exists(), "full response child creates its own output directory"
            folder.mkdir()
            cases = [{"name": f"fixture_{case}", "exact": True,
                      "epochs": [{"epoch": epoch, "exact": True} for epoch in range(3)]}
                     for case in range(11)]
            (folder / "response_epoch.json").write_text(json.dumps({"exact": True, "cases": cases}))
        return SimpleNamespace(returncode=0)

    report = run_bundle(tmp_path / "bundle", runner=runner)
    assert report["all_exact"]
    assert [row["name"] for row in report["steps"]] == ["packing", "composition", "full_response"]
    assert len(calls) == 3


def test_followup_does_not_accept_short_full_response_report(tmp_path):
    from benchmarks.beam_response_followup_bundle import run_bundle

    def runner(command, **_):
        folder = Path(command[-1])
        if "--stage" in command:
            stage = command[command.index("--stage") + 1]
            (folder / "probe.json").write_text(json.dumps({"stage": stage, "compiled": True}))
            (folder / "lowered.mlir").write_text("lowered")
            (folder / "compiled.hlo.txt").write_text("hlo")
        else:
            folder.mkdir()
            (folder / "response_epoch.json").write_text(json.dumps({"exact": True, "cases": []}))
        return SimpleNamespace(returncode=0)

    report = run_bundle(tmp_path / "bundle", runner=runner)
    assert not report["all_exact"]
    assert not report["steps"][-1]["exact"]
