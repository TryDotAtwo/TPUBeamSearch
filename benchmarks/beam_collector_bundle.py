"""Sequential isolated collector gates. Coordinator owns no JAX client."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_bundle(output,*,runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report = {'groups':[],'all_exact':False,
              'scope':'isolated functional collectors, not matched A/B or full beam'}
    def save():
        (output/'collector_bundle.json').write_text(json.dumps(report,indent=2))
    save()
    for name,module,flags,filename in (
        ('single','benchmarks.beam_collector_probe',(),'collector.json'),
        ('group','benchmarks.beam_collector_probe',('--group',),'collector.json'),
        ('full','benchmarks.beam_collector_full_probe',(),'collector_full.json')):
        folder = output/name
        folder.mkdir(parents=True,exist_ok=True)
        row = {'name':name,'exact':False,'status':'running'}
        report['groups'].append(row)
        save()
        print('GROUP',name,flush=True)
        with (folder/'process.log').open('w',encoding='utf-8') as log:
            result = runner([sys.executable,'-m',module,'--output',str(folder),*flags],
                            stdout=log,stderr=subprocess.STDOUT,check=False)
        row.update(returncode=result.returncode,status='returned')
        path = folder/filename
        if path.exists():
            try:
                nested = json.loads(path.read_text())
                row['report'] = nested
                exact = nested.get('exact') is True if name == 'full' else (
                    len(nested.get('cases',[])) == 1 and nested['cases'][0].get('exact') is True)
                row['exact'] = result.returncode == 0 and exact
            except (ValueError,OSError) as error:
                row['report_error'] = str(error)
        save()
        print('GROUP_DONE',name,result.returncode,flush=True)
    report['all_exact'] = all(x['exact'] for x in report['groups'])
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    result = run_bundle(parser.parse_args().output)
    raise SystemExit(0 if result['all_exact'] else 1)
