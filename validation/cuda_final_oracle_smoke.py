"""Full-byte CUDA materialization oracle fixtures; not TPU or beam acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(609)
    parents=rng.integers(0,120,(7,128),dtype=np.uint8)
    parents[:,120:]=0
    generators=np.tile(np.arange(128,dtype=np.uint8),(24,1))
    for move in range(24):
        generators[move,:120]=np.roll(np.arange(120,dtype=np.uint8),move)
    report={'scope':'one physical CUDA GPU, original materialize/validation; independent NumPy byte reference, not TPU or full beam','cases':[]}
    for count,local in [(n,False) for n in (0,1,127,128,129)]+[(127,True)]:
        requests=[]
        expected=np.zeros((count,128),np.uint8)
        for index in range(count):
            parent=index%7
            move=index%24
            target=index if local else count-1-index
            requests.append(struct.pack('<QIHBx',parent,target,index%8,move))
            expected[index,:120]=parents[parent,generators[move,:120]]
            expected[index,120:124]=np.frombuffer(struct.pack('<I',target),np.uint8)
        blob=b'TFIN0001'+struct.pack('<6I',7,count,24,120,128,count)+parents.tobytes()+b''.join(requests)+generators.tobytes()
        label=f'count{count}_'+('local' if local else 'remote')
        fixture=args.output/f'{label}.bin'
        result=args.output/f'{label}.out'
        fixture.write_bytes(blob)
        run=subprocess.run([str(args.oracle.resolve()),str(fixture.resolve()),str(result.resolve()),*(['--local-slots'] if local else [])],capture_output=True,text=True,check=False)
        actual=result.read_bytes() if result.exists() else b''
        exact=run.returncode==0 and actual==expected.tobytes()
        report['cases'].append(dict(count=count,mode='local' if local else 'remote',returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,
            input_sha256=hashlib.sha256(blob).hexdigest(),expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            output_sha256=hashlib.sha256(actual).hexdigest(),exact=exact))
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        if not exact:
            raise RuntimeError(f'CUDA case count{count} failed')
    report['all_exact']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
