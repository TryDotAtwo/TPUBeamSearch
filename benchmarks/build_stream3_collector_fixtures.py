"""Original CPU C++ S3/routing plus source-audited Python admission, not CUDA."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from benchmarks.build_external_stream3_fixtures import query_oracle, sha
from tpu_beam_search.beam_collector import reserve_group


def reference_admit(exe,a,b,controls,records):
    shards,_,capacity = a.shape
    tokens = [records.shape[1],8,shards,*records[:4].T.ravel()]
    result = subprocess.run([str(exe),'route'],input=' '.join(map(str,tokens)),
                            text=True,capture_output=True,check=True,timeout=30)
    routing = np.asarray([list(map(int,x.split())) for x in result.stdout.splitlines()],np.uint32).reshape(-1,2)
    if len(routing) != records.shape[1]:
        raise ValueError('route oracle count mismatch')
    groups = [records[:,routing[:,1] == s] for s in range(shards)]
    plans = [reserve_group(capacity=capacity,
        clean=tuple(int(x) for x in controls[s,:2,0]),
        dirty=tuple(int(x) for x in controls[s,2:4,0]),
        processing=tuple(int(x) for x in controls[s,4:6,0]),
        current=int(controls[s,6,0]),amount=g.shape[1]) for s,g in enumerate(groups)]
    failed = bool(np.any(controls[:,7,0])) or any(p.fatal_overflow for p in plans)
    aa,bb,cc = a.copy(),b.copy(),controls.copy()
    cc[:,7,0] = failed
    if not failed:
        for s,(p,g) in enumerate(zip(plans,groups)):
            if p.buffer is not None:
                (aa if p.buffer == 0 else bb)[s,:,p.offset:p.offset+g.shape[1]] = g
                cc[s,2:4,0] = p.dirty
                cc[s,6,0] = p.buffer
    return aa,bb,cc,failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(9851)
    words = rng.integers(0,2**32,(8,8,128),dtype=np.uint32)
    words[:,:4] = 0
    words[:,0] = rng.integers(0,64,(8,128),dtype=np.uint32)
    words[:,3] = np.uint32(0x80000000)
    words[:,6] = rng.integers(0,8,(8,128),dtype=np.uint32)
    words[:,7] %= 24
    payload = np.broadcast_to(np.arange(127,-1,-1,dtype=np.uint32),(8,1,128)).copy()
    counts = np.asarray([0,128,127,1,128,70,128,128],np.uint32)[:,None]
    thresholds = np.asarray([5,5,0,0xffffffff,7,5,3,5],np.uint32)[:,None]
    a = rng.integers(0,2**32,(8,3,8,256),dtype=np.uint32)
    b = rng.integers(0,2**32,a.shape,dtype=np.uint32)
    controls = np.zeros((8,3,8,128),np.uint32)
    controls[1,:,0,0] = 250
    controls[3,:,:2,0] = 256
    controls[4,:,4,0] = 1
    controls[5,:,7,0] = 1
    controls[6,:,:2,0] = 240
    neutral = np.zeros((8,8,128),np.uint32)
    neutral[:,6] = np.uint32(0xffffffff)
    split = [query_oracle(args.oracle,words[r],payload[r],int(counts[r,0]),
                           int(thresholds[r,0]),r) for r in range(8)]
    expected = []
    receive_counts = []
    for rank in range(8):
        local,_,lc,_,_ = split[rank]
        aa,bb,cc,_ = reference_admit(args.oracle,a[rank],b[rank],controls[rank],local[:,:int(lc[0,0])])
        remote_groups = []
        per_peer = []
        for source in range(8):
            _,remote,_,send_count,send_offset = split[source]
            count,start = int(send_count[0,rank]),int(send_offset[0,rank])
            remote_groups.append(remote[:,start:start+count])
            per_peer.append(count)
        receive_counts.append(per_peer)
        aa,bb,cc,fatal = reference_admit(args.oracle,aa,bb,cc,np.concatenate(remote_groups,axis=1))
        flag = np.zeros((1,128),np.uint32)
        flag[0,0] = fatal
        expected.append((aa,bb,cc,flag))
    path = args.output/'stream3_collector_128.npz'
    np.savez_compressed(path,a=a,b=b,controls=controls,words=words,payload=payload,
        counts=counts,thresholds=thresholds,neutral=neutral,
        **{f'expected_{i}':np.stack([x[i] for x in expected]) for i in range(4)})
    manifest = dict(scope='CPU original C++ S3/routing and source-audited Python admission; not CUDA',
        source_commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip()),
        source_files={},adapter_sha256=sha(Path('tests/beam_source_oracle.cpp')),
        executable_sha256=sha(args.oracle),file=path.name,sha256=sha(path),
        seed=9851,receive_counts=receive_counts,
        expected_fatal=[int(x[3][0,0]) for x in expected])
    for relative in ('src/hash.hpp','src/hash.cpp','src/stream3.hpp','src/stream3.cpp',
                     'src/types.hpp','src/config.hpp','cuda/stream3.cu','cuda/dispatcher.cu'):
        manifest['source_files'][relative] = sha(args.source/relative)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
