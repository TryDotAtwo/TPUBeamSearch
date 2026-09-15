"""No-JAX, sequential compile isolation for the response epoch V4 abort."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

STAGES = ('packing', 'packing_control', 'packing_selection',
          'packing_guard', 'packing_first_dma', 'packing_second_dma', 'packing_gather',
          'exchange', 'receive', 'planes_to_wire', 'composition')


def run_bundle(output, *, runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(all_compiled=False, cases=[],
                  scope='compile isolation only; no correctness or timing acceptance')

    def save():
        (output/'response_isolation.json').write_text(json.dumps(report, indent=2))

    save()
    for stage in STAGES:
        folder = output/stage
        folder.mkdir(exist_ok=False)
        row = dict(stage=stage, status='running', compiled=False)
        report['cases'].append(row)
        save()
        with (folder/'process.log').open('w', encoding='utf-8') as log:
            result = runner([sys.executable, '-m', 'benchmarks.beam_response_isolation_probe',
                             '--stage', stage, '--output', str(folder)],
                            stdout=log, stderr=subprocess.STDOUT, check=False)
        row.update(returncode=result.returncode, status='returned')
        try:
            nested = json.loads((folder/'probe.json').read_text())
            row['report'] = nested
            row['compiled'] = (result.returncode == 0 and nested.get('stage') == stage
                               and nested.get('compiled') is True
                               and (folder/'lowered.mlir').is_file()
                               and (folder/'compiled.hlo.txt').is_file())
        except (OSError, ValueError, AttributeError) as error:
            row['report_error'] = str(error)
        save()
    report['all_compiled'] = all(row['compiled'] for row in report['cases'])
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(0 if run_bundle(parser.parse_args().output)['all_compiled'] else 1)
