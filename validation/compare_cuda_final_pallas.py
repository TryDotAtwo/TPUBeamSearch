"""Actual CUDA bytes versus Pallas interpretation of identical saved fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import jax.numpy as jnp
import numpy as np
from tpu_beam_search.beam_final_materialize import pallas_materialize_final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',type=Path,required=True)
    args=parser.parse_args()
    source=json.loads((args.fixtures/'report.json').read_text())
    if source.get('all_exact') is not True:
        raise RuntimeError('CUDA oracle must pass first')
    report={'scope':'actual single GPU output vs Pallas CPU interpreter, NOT physical TPU','cases':[]}
    for row in source['cases']:
        count=row['count']
        label=f"count{count}_{row['mode']}"
        blob=(args.fixtures/f'{label}.bin').read_bytes()
        if hashlib.sha256(blob).hexdigest()!=row['input_sha256'] or blob[:8]!=b'TFIN0001':
            raise RuntimeError('fixture provenance mismatch')
        parents_count,n,moves,state_len,width,target_count=struct.unpack_from('<6I',blob,8)
        if n!=count:
            raise RuntimeError('count mismatch')
        cursor=32
        parents=np.frombuffer(blob,np.uint8,count=parents_count*width,offset=cursor).reshape(parents_count,width).copy()
        cursor+=parents_count*width
        raw=np.frombuffer(blob,'<u4',count=count*4,offset=cursor).reshape(count,4)
        cursor+=count*16
        generators=np.frombuffer(blob,np.uint8,count=moves*width,offset=cursor).reshape(moves,width).astype(np.int32)
        capacity=max(128,((count+127)//128)*128)
        requests=np.zeros((4,capacity),np.uint32)
        requests[:,:count]=raw.T
        wire,errors=pallas_materialize_final(*map(jnp.asarray,(parents,generators,requests)),
            jnp.array([count],jnp.uint32),jnp.array([target_count],jnp.uint32),state_len=state_len,interpret=True)
        actual=np.asarray(wire)
        cuda=(args.fixtures/f'{label}.out').read_bytes()
        if hashlib.sha256(cuda).hexdigest()!=row['output_sha256']:
            raise RuntimeError('CUDA output provenance mismatch')
        exact=int(errors[0,0])==0 and actual[:count].tobytes()==cuda and not actual[count:].any()
        report['cases'].append(dict(name=label,input_sha256=row['input_sha256'],cuda_sha256=row['output_sha256'],
            pallas_sha256=hashlib.sha256(actual[:count].tobytes()).hexdigest(),exact=bool(exact)))
        (args.fixtures/'pallas_interpreter_comparison.json').write_text(json.dumps(report,indent=2))
        if not exact:
            raise RuntimeError(f'{label} mismatch')
    report['all_exact']=True
    (args.fixtures/'pallas_interpreter_comparison.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
