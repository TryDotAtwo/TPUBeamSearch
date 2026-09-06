"""Sequential independent S4/S5 gates; coordinator creates no JAX client."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_bundle(output,*,runner=subprocess.run,recovery=False,transport=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(groups=[],all_exact=False,scope='independent primitives; not integrated S5 or beam')
    def save():
        (output/'s4_s5_bundle.json').write_text(json.dumps(report,indent=2))
    save()
    groups = (
        ('s4','benchmarks.beam_s4_probe',(),'s4_reserved.json'),
        ('request','benchmarks.beam_s5_request_probe',('--kind','request'),'s5_request.json'),
        ('histogram','benchmarks.beam_s5_request_probe',('--kind','histogram'),'s5_histogram.json'))
    if recovery:
        groups = tuple((name,'benchmarks.beam_s5_request_probe',('--kind',name),f's5_{name}.json')
                       for name in ('request','wire','reduction','combined'))
    if transport:
        groups = tuple((name,'benchmarks.beam_s5_request_probe',('--kind',name),f's5_{name}.json')
                       for name in ('own','wire'))
    for name,module,flags,filename in groups:
        folder = output/name
        folder.mkdir(parents=True,exist_ok=True)
        row = dict(name=name,exact=False,status='running')
        report['groups'].append(row)
        save()
        print('GROUP',name,flush=True)
        with (folder/'process.log').open('w',encoding='utf-8') as log:
            result = runner([sys.executable,'-m',module,'--output',str(folder),*flags],
                            stdout=log,stderr=subprocess.STDOUT,check=False)
        row.update(returncode=result.returncode,status='returned')
        try:
            nested = json.loads((folder/filename).read_text())
            row['report'] = nested
            row['exact'] = result.returncode == 0 and isinstance(nested,dict) and nested.get('exact') is True
        except (OSError,ValueError) as error:
            row['report_error'] = str(error)
        save()
    report['all_exact'] = all(r['exact'] for r in report['groups'])
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--recovery',action='store_true')
    parser.add_argument('--transport',action='store_true')
    args = parser.parse_args()
    result = run_bundle(args.output,recovery=args.recovery,transport=args.transport)
    raise SystemExit(0 if result['all_exact'] else 1)
