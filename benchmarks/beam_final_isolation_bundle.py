"""No-JAX coordinator for independent native-abort isolation subprocesses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

MODES = ('dma','cast','gather_1d','gather_2d','packing','production')


def run_bundle(output, *, runner=subprocess.run):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(all_exact=False,cases=[],scope='compile isolation, not full beam')
    def save():
        (output/'isolation_bundle.json').write_text(json.dumps(report,indent=2))
    save()
    for mode in MODES:
        folder = output/mode
        folder.mkdir(exist_ok=True)
        row = dict(mode=mode,exact=False,status='running')
        report['cases'].append(row)
        save()
        with (folder/'process.log').open('w',encoding='utf-8') as log:
            result = runner([sys.executable,'-m','benchmarks.beam_final_isolation_probe',
                             '--mode',mode,'--output',str(folder)],
                            stdout=log,stderr=subprocess.STDOUT,check=False)
        row.update(returncode=result.returncode,status='returned')
        try:
            nested = json.loads((folder/'probe.json').read_text())
            row['report'] = nested
            row['exact'] = (result.returncode == 0 and nested.get('mode') == mode
                            and nested.get('exact') is True)
        except (OSError,ValueError,AttributeError) as error:
            row['report_error'] = str(error)
        save()
    report['all_exact'] = all(row['exact'] for row in report['cases'])
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if run_bundle(parser.parse_args().output)['all_exact'] else 1)
