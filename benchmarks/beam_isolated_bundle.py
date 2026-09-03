"""Sequential process isolation for fatal native TPU compiler failures.

The coordinator intentionally imports no JAX and owns no TPU client. All four
packing variants stay together to preserve their matched interleaved timing.
Other groups are diagnostic only; cross-process timings are not matched A/B.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


GROUPS = {
    'pack_matched': ['pack_serial_b2', 'pack_serial_b3',
                     'pack_pipeline_b2', 'pack_pipeline_b3'],
    'routing': ['route_8_7'],
    'hash_120': ['hash_goal_120_24'],
    'hash_150': ['hash_goal_150_30'],
    **{f'dedup_{mode}_{n}': [f'dedup_{mode}_{n}']
       for n in (128, 256) for mode in ('stream3', 'stream4')},
}


def run_groups(output, groups=GROUPS, *, runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'sequential isolated eight-TPU primitive groups; not full beam',
              'groups': [], 'all_exact': False}
    path = output / 'isolated_bundle.json'

    def save():
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    for name, cases in groups.items():
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'benchmarks.beam_primitive_bundle',
                   '--output', str(folder)]
        for case in cases:
            command.extend(('--case', case))
        row = {'name': name, 'requested_cases': cases, 'status': 'running'}
        report['groups'].append(row)
        save()
        print('GROUP', name, flush=True)
        with (folder / 'process.log').open('w', encoding='utf-8') as log:
            result = runner(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        row['returncode'] = result.returncode
        row['status'] = 'returned' if result.returncode == 0 else 'process_error'
        result_path = folder / 'beam_primitives.json'
        if result_path.exists():
            try:
                row['report'] = json.loads(result_path.read_text(encoding='utf-8'))
            except (ValueError, OSError) as error:
                row['report_error'] = str(error)
        save()
        print('GROUP_DONE', name, result.returncode, flush=True)
    report['all_exact'] = bool(report['groups']) and all(
        r['returncode'] == 0 and r.get('report', {}).get('all_exact') is True
        and sorted(c['name'] for c in r['report'].get('cases', [])) == sorted(r['requested_cases'])
        and all(c.get('status') == 'exact' for c in r['report']['cases'])
        for r in report['groups'])
    save()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_groups(args.output)


if __name__ == '__main__':
    main()
