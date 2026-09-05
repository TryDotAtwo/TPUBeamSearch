"""Generate replay fixtures from the original CPU C++ Stream3, not CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def query_oracle(exe, words, payload, count, threshold, rank):
    records = np.concatenate((words,payload))[:, :count]
    tokens = [count,threshold,rank,8,*records.T.ravel()]
    raw = subprocess.run([str(exe),'stream3'],input=' '.join(map(str,tokens)),
                         text=True,capture_output=True,check=True,timeout=30)
    lines = raw.stdout.splitlines()
    if lines[0] != 'STREAM3':
        raise ValueError('unexpected oracle format')
    local_n, remote_n = map(int,lines[1].split())
    rows = np.array([list(map(int,x.split())) for x in lines[4:]],np.uint32).reshape(-1,8)
    if len(rows) != local_n+remote_n:
        raise ValueError('oracle count mismatch')
    outputs = []
    for group in (rows[:local_n],rows[local_n:]):
        out = np.zeros_like(words)
        out[6] = np.uint32(0xffffffff)
        out[:,:len(group)] = group.T
        outputs.append(out)
    lc, counts, offsets = (np.zeros((1,128),np.uint32) for _ in range(3))
    lc[0,0] = local_n
    counts[0,:8] = list(map(int,lines[2].split()))
    offsets[0,:9] = list(map(int,lines[3].split()))
    return (*outputs,lc,counts,offsets)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = dict(scope='CPU original Stream3 oracle; not CUDA',
        source_commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip()),
        adapter_sha256=sha(Path('tests/beam_source_oracle.cpp')),
        executable_sha256=sha(args.oracle),source_files={},cases=[])
    for path in sorted((args.source/'src').glob('*.hpp')):
        manifest['source_files'][path.name] = sha(path)
    for name in ('hash.cpp','state.cpp','stream3.cpp','stream4.cpp'):
        manifest['source_files'][name] = sha(args.source/'src'/name)
    for n in (256,512):
        rng = np.random.default_rng(6581+n)
        words = rng.integers(0,2**32,(8,8,n),dtype=np.uint32)
        words[:, :4] = 0
        words[:,0] = rng.integers(0,n//2,(8,n),dtype=np.uint32)
        words[:,3] = np.uint32(0x80000000)
        words[:,6] = rng.integers(0,8,(8,n),dtype=np.uint32)
        words[:,7] %= 24
        words[2,:4] = 0
        words[7,6,0] = np.uint32(0xffffffff)
        payload = np.broadcast_to(np.arange(n-1,-1,-1,dtype=np.uint32),(8,1,n)).copy()
        counts = np.array([0,n,n,n-1,129,128,1,n],np.uint32)[:,None,None]
        thresholds = np.array([5,5,5,0,7,5,0,0xffffffff],np.uint32)[:,None,None]
        expected = list(zip(*[query_oracle(args.oracle,words[r],payload[r],
            int(counts[r,0,0]),int(thresholds[r,0,0]),r) for r in range(8)]))
        path = args.output/f'stream3_{n}.npz'
        np.savez_compressed(path,words=words,payload=payload,counts=counts,thresholds=thresholds,
                            **{f'expected_{i}':np.stack(x) for i,x in enumerate(expected)})
        manifest['cases'].append(dict(capacity=n,file=path.name,sha256=sha(path),seed=6581+n))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
