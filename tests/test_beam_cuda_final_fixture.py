from pathlib import Path
import hashlib
import numpy as np
import json
import shutil
import struct
import pytest


def test_load_published_cuda_bytes_and_preserve_request_words():
    from benchmarks.beam_cuda_final_fixture import load_cases
    root=Path(__file__).resolve().parents[1]/'test_results/cuda_final_oracle_v2'
    cases=load_cases(root)
    assert len(cases)==6
    for case in cases:
        parents,generators,requests,count,target=case['inputs']
        n=int(count[0])
        assert parents.shape==(7,128) and generators.shape==(24,128)
        assert requests.shape[0]==4 and requests.shape[1]%128==0
        assert case['expected'].shape==(requests.shape[1],128)
        assert not case['expected'][n:].any()
        assert hashlib.sha256(case['expected'][:n].tobytes()).hexdigest()==case['cuda_sha256']
        if n:
            np.testing.assert_array_equal(requests[0,:n],np.arange(n)%7)
            np.testing.assert_array_equal(requests[3,:n]>>16,np.arange(n)%24)
        assert int(target[0])==n


@pytest.mark.parametrize('damage', ['hash', 'trailing', 'count', 'duplicate', 'output', 'failed'])
def test_reject_corrupt_cuda_evidence(tmp_path, damage):
    from benchmarks.beam_cuda_final_fixture import load_cases
    root = Path(__file__).resolve().parents[1] / 'test_results/cuda_final_oracle_v2'
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    report_path = tmp_path / 'report.json'
    report = json.loads(report_path.read_text())
    row = report['cases'][0]
    path = tmp_path / 'count0_remote.bin'
    blob = bytearray(path.read_bytes())
    if damage == 'hash':
        blob[-1] ^= 1
    elif damage == 'trailing':
        blob.append(0)
        row['input_sha256'] = hashlib.sha256(blob).hexdigest()
    elif damage == 'count':
        struct.pack_into('<I', blob, 12, 1)
        row['input_sha256'] = hashlib.sha256(blob).hexdigest()
    elif damage == 'duplicate':
        report['cases'].append(row.copy())
    elif damage == 'output':
        (tmp_path / 'count0_remote.out').write_bytes(b'bad')
    elif damage == 'failed':
        row['returncode'] = 1
    path.write_bytes(blob)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_cases(tmp_path)
