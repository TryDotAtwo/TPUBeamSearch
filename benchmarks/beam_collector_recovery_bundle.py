"""Retry full collector, then conditionally exercise integrated S3 in one session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_bundle(output,*,runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(groups=[],all_exact=False,scope='full collector then conditional integrated gate; no matched speed claim')
    def save():
        (output/'recovery_bundle.json').write_text(json.dumps(report,indent=2))
    save()
    for name,module,filename in (
        ('full','benchmarks.beam_collector_full_probe','collector_full.json'),
        ('integrated','benchmarks.beam_stream3_collector_probe','stream3_collector.json')):
        folder = output/name
        folder.mkdir(parents=True,exist_ok=True)
        row = dict(name=name,exact=False,status='running')
        report['groups'].append(row)
        save()
        with (folder/'process.log').open('w',encoding='utf-8') as log:
            result = runner([sys.executable,'-m',module,'--output',str(folder)],
                            stdout=log,stderr=subprocess.STDOUT,check=False)
        row.update(returncode=result.returncode,status='returned')
        try:
            nested = json.loads((folder/filename).read_text())
            row['report'] = nested
            row['exact'] = result.returncode == 0 and nested.get('exact') is True
        except (OSError,ValueError) as error:
            row['report_error'] = str(error)
        save()
        if not row['exact']:
            return report
    report['all_exact'] = True
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    result = run_bundle(parser.parse_args().output)
    raise SystemExit(0 if result['all_exact'] else 1)
