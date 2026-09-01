"""Extract compact MXU schedule records from TPU compiled-HLO text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_LIST_FIELDS = (
    "iteration_bounds",
    "input_window_bounds",
    "kernel_window_bounds",
    "output_window_bounds",
)


def _integer_list(line: str, field: str) -> list[int]:
    match = re.search(rf'"{re.escape(field)}":\[(.*?)\]', line)
    if match is None or not match.group(1):
        return []
    return [int(value) for value in re.findall(r'"(-?\d+)"', match.group(1))]


def extract_mxu_schedules(text: str) -> list[dict]:
    """Return ordered convolution-emitter/window metadata without parsing all HLO."""
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if '"convolution_algorithm_config"' not in line:
            continue
        op_name = re.search(r'op_name="([^"]+)"', line)
        emitter = re.search(r'"emitter":"([^"]+)"', line)
        row = dict(
            index=len(rows),
            line=line_number,
            op_name=op_name.group(1) if op_name else None,
            emitter=emitter.group(1) if emitter else None,
        )
        row.update({field: _integer_list(line, field) for field in _LIST_FIELDS})
        scoped = line.split('"window_config"', 1)[0]
        row["scoped_memory_bytes"] = [
            int(value) for value in re.findall(r'"size":"(\d+)"', scoped)
        ]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    result = {
        str(path): extract_mxu_schedules(path.read_text(encoding="utf-8"))
        for path in args.files
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
