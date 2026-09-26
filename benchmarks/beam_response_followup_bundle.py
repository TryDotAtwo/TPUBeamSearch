"""Run response compile isolation before the full 11-fixture TPU gate."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_bundle(output, *, runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"all_exact": False, "steps": [],
              "scope": "two compile-isolation subprocesses, then 11 fixtures x 3 epochs"}
    manifest = output / "followup.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    for stage in ("packing", "composition", "full_response"):
        folder = output / stage
        if stage != "full_response":
            folder.mkdir(exist_ok=False)
        row = {"name": stage, "status": "running", "exact": False}
        report["steps"].append(row)
        save()
        command = [sys.executable, "-m",
                   "benchmarks.beam_response_epoch_probe" if stage == "full_response"
                   else "benchmarks.beam_response_isolation_probe"]
        if stage != "full_response":
            command.extend(("--stage", stage))
        command.extend(("--output", str(folder)))
        log_path = (output / "full_response.process.log" if stage == "full_response"
                    else folder / "process.log")
        with log_path.open("w", encoding="utf-8") as log:
            result = runner(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row.update(status="returned", returncode=result.returncode)
        nested_name = "response_epoch.json" if stage == "full_response" else "probe.json"
        try:
            nested = json.loads((folder / nested_name).read_text(encoding="utf-8"))
            row["report"] = nested
            if stage == "full_response":
                cases = nested.get("cases", [])
                row["exact"] = bool(result.returncode == 0 and nested.get("exact") is True
                                    and len(cases) == 11
                                    and len({case.get("name") for case in cases}) == 11
                                    and all(case.get("exact") is True
                                            and [item.get("epoch") for item in case.get("epochs", [])] == [0, 1, 2]
                                            and all(item.get("exact") is True for item in case["epochs"])
                                            for case in cases))
            else:
                row["exact"] = bool(result.returncode == 0 and nested.get("stage") == stage
                                    and nested.get("compiled") is True
                                    and (folder / "lowered.mlir").is_file()
                                    and (folder / "compiled.hlo.txt").is_file())
        except (OSError, ValueError, AttributeError, TypeError) as error:
            row["report_error"] = str(error)
        save()
        if not row["exact"]:
            return report
    report["all_exact"] = True
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run_bundle(parser.parse_args().output)["all_exact"] else 1)
