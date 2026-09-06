"""Sequential final gates; the coordinator never initializes a JAX client."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_bundle(output, *, runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(all_exact=False, groups=[], scope='materialization and exchange separately; not full beam')

    def save():
        (output / 'final_bundle.json').write_text(json.dumps(report, indent=2))

    save()
    for name, module, gate, count in (
        ('cuda_final', 'benchmarks.beam_cuda_final_probe', 'all_exact', 6),
        ('final_exchange', 'benchmarks.beam_final_exchange_probe', 'exact', 16),
    ):
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(name=name, exact=False, status='running')
        report['groups'].append(row)
        save()
        print('GROUP', name, flush=True)
        with (folder / 'process.log').open('w', encoding='utf-8') as log:
            result = runner([sys.executable, '-m', module, '--output', str(folder)],
                            stdout=log, stderr=subprocess.STDOUT, check=False)
        row.update(returncode=result.returncode, status='returned')
        try:
            nested = json.loads((folder / f'{name}.json').read_text())
            row['report'] = nested
            cases = nested.get('cases', [])
            row['exact'] = (result.returncode == 0 and nested.get(gate) is True
                            and len(cases) == count and all(x.get('exact') is True for x in cases)
                            and len({x['name'] for x in cases}) == count)
        except (OSError, ValueError, AttributeError, KeyError, TypeError) as error:
            row['report_error'] = str(error)
        save()
    report['all_exact'] = all(row['exact'] for row in report['groups'])
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    result = run_bundle(parser.parse_args().output)
    raise SystemExit(0 if result['all_exact'] else 1)
