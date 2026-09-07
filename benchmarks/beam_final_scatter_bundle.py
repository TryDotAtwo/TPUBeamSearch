"""No-JAX isolated scatter gates with independently reconstructed acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .beam_final_scatter_fixture import make_case

MODES = ('scatter', 'integrated')
RUNTIME_FIELDS = ('jax', 'jaxlib', 'libtpu')


def validate_report(report, mode, source_sha):
    errors = []
    if not isinstance(report, dict):
        return ['report is not an object']
    for field, expected in (('mode', mode), ('source_sha', source_sha), ('exact', True)):
        if report.get(field) != expected or (field == 'exact' and report.get(field) is not True):
            errors.append(f'invalid {field}')
    for field in RUNTIME_FIELDS:
        if not isinstance(report.get(field), str) or not report[field].strip():
            errors.append(f'missing runtime {field}')
    devices = report.get('devices')
    if (not isinstance(devices, list) or len(devices) != 8
            or any(not isinstance(d, dict) or type(d.get('id')) is not int
                   or not str(d.get('kind', '')).startswith('TPU') for d in devices)):
        errors.append('requires eight physical TPU device records')
    elif len({d['id'] for d in devices}) != 8:
        errors.append('duplicate devices')
    fixtures = [make_case(n) for n in (0, 1, 127, 128, 129)]
    if mode == 'scatter':
        fixtures += [make_case(129, failure=f) for f in ('count_overflow', 'target_overflow')]
    rows = report.get('cases')
    if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
            or sorted(str(r.get('name')) for r in rows) != sorted(c['name'] for c in fixtures)):
        return errors + ['wrong case identities']
    by_name = {r['name']: r for r in rows}
    for case in fixtures:
        row = by_name[case['name']]
        digest = hashlib.sha256(case['expected'].tobytes()).hexdigest()
        inputs_digest = hashlib.sha256(b''.join(x.tobytes() for x in case['inputs'])).hexdigest()
        if (row.get('exact') is not True or row.get('status') != 'executed'
                or row.get('input_sha256') != inputs_digest
                or row.get('expected_sha256') != digest
                or row.get('output_sha256') != [digest]*8
                or type(row.get('expected_error')) is not int
                or row.get('expected_error') != case['expected_error']):
            errors.append(f"invalid identity/hash/status: {case['name']}")
        for field, expected in (('mismatches', 0), ('invalid', case['expected_error'])):
            values = row.get(field)
            if (not isinstance(values, list) or len(values) != 8
                    or any(type(v) is not int or v != expected for v in values)):
                errors.append(f"invalid {field}: {case['name']}")
    return errors


def run_bundle(output, *, source_sha, runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(all_exact=False, source_sha=source_sha, cases=[],
                  scope='local scatter/chain gates, not distributed final publication or timing')
    def save():
        (output/'scatter_bundle.json').write_text(json.dumps(report, indent=2))
    save()
    for mode in MODES:
        folder = output/mode
        folder.mkdir()
        row = dict(mode=mode, exact=False, status='running')
        report['cases'].append(row)
        save()
        with (folder/'process.log').open('w', encoding='utf-8') as log:
            result = runner([sys.executable, '-m', 'benchmarks.beam_final_scatter_probe',
                             '--mode', mode, '--output', str(folder)],
                            stdout=log, stderr=subprocess.STDOUT, check=False)
        row.update(returncode=result.returncode, status='returned')
        try:
            nested = json.loads((folder/'probe.json').read_text())
            row['report'] = nested
            row['validation_errors'] = validate_report(nested, mode, source_sha)
            row['exact'] = result.returncode == 0 and not row['validation_errors']
        except (OSError, ValueError) as error:
            row['report_error'] = str(error)
        save()
    if all(row['exact'] for row in report['cases']):
        left, right = (r['report'] for r in report['cases'])
        report['runtime_matches'] = all(left[k] == right[k] for k in RUNTIME_FIELDS)
        report['all_exact'] = report['runtime_matches']
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run_bundle(args.output, source_sha=args.source_sha)['all_exact'] else 1)
